"""
Enterprise Internal Link Builder
A scalable tool for building internal links at scale (30-50k+ pages)
Features:
- Async crawling with Googlebot user agent
- SQLite storage for persistence
- Semantic embeddings with sentence-transformers
- UMAP/t-SNE visualization
- HDBSCAN clustering for topic detection
- AI-powered link suggestions
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from aiohttp import ClientTimeout
import sqlite3
from datetime import datetime
import json
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import time
import io
import base64

# Lazy imports for heavy ML libraries
def get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_umap():
    import umap
    return umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)

def get_hdbscan():
    import hdbscan
    return hdbscan

def get_sklearn():
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    return cosine_similarity, KMeans

# Page config
st.set_page_config(
    page_title="Internal Link Builder Pro",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Database setup
def init_db():
    conn = sqlite3.connect('internal_links.db', check_same_thread=False)
    c = conn.cursor()
    
    # Pages table
    c.execute('''CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        domain TEXT,
        meta_title TEXT,
        meta_description TEXT,
        h1 TEXT,
        body_content TEXT,
        word_count INTEGER,
        internal_links_count INTEGER,
        external_links_count INTEGER,
        crawl_status TEXT,
        crawl_error TEXT,
        crawled_at TIMESTAMP,
        embedding BLOB,
        cluster_id INTEGER,
        page_rank REAL DEFAULT 0.0
    )''')
    
    # Internal links table
    c.execute('''CREATE TABLE IF NOT EXISTS internal_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT,
        target_url TEXT,
        anchor_text TEXT,
        context TEXT,
        is_dofollow INTEGER DEFAULT 1,
        UNIQUE(source_url, target_url, anchor_text)
    )''')
    
    # Suggestions table
    c.execute('''CREATE TABLE IF NOT EXISTS suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT,
        target_url TEXT,
        suggested_anchor TEXT,
        suggested_context TEXT,
        relevance_score REAL,
        action TEXT,
        priority TEXT,
        reason TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Clusters table
    c.execute('''CREATE TABLE IF NOT EXISTS clusters (
        id INTEGER PRIMARY KEY,
        name TEXT,
        description TEXT,
        page_count INTEGER,
        pillar_url TEXT
    )''')
    
    conn.commit()
    return conn

# User agents
USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_mobile': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'custom': ''
}

# Async crawler class
class AsyncCrawler:
    def __init__(self, user_agent='googlebot', max_concurrent=10, timeout=30, delay=0.5):
        self.user_agent = USER_AGENTS.get(user_agent, user_agent)
        self.max_concurrent = max_concurrent
        self.timeout = ClientTimeout(total=timeout)
        self.delay = delay
        self.semaphore = None
        
    async def fetch_page(self, session, url, progress_callback=None):
        """Fetch a single page"""
        try:
            if self.semaphore:
                async with self.semaphore:
                    await asyncio.sleep(self.delay)  # Rate limiting
                    async with session.get(url, timeout=self.timeout, allow_redirects=True) as response:
                        if response.status == 200:
                            html = await response.text()
                            return {'url': url, 'html': html, 'status': 'success', 'final_url': str(response.url)}
                        else:
                            return {'url': url, 'status': 'error', 'error': f'HTTP {response.status}'}
        except asyncio.TimeoutError:
            return {'url': url, 'status': 'error', 'error': 'Timeout'}
        except Exception as e:
            return {'url': url, 'status': 'error', 'error': str(e)}
    
    def parse_page(self, html, url):
        """Parse HTML and extract data"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, nav, footer elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            element.decompose()
        
        # Extract meta data
        meta_title = ''
        title_tag = soup.find('title')
        if title_tag:
            meta_title = title_tag.get_text(strip=True)
        
        meta_description = ''
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_desc_tag:
            meta_description = meta_desc_tag.get('content', '')
        
        h1 = ''
        h1_tag = soup.find('h1')
        if h1_tag:
            h1 = h1_tag.get_text(strip=True)
        
        # Extract body content (main article content)
        body_content = ''
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post|entry'))
        if main_content:
            body_content = main_content.get_text(separator=' ', strip=True)
        else:
            body = soup.find('body')
            if body:
                body_content = body.get_text(separator=' ', strip=True)
        
        # Clean body content
        body_content = re.sub(r'\s+', ' ', body_content)[:10000]  # Limit to 10k chars
        word_count = len(body_content.split())
        
        # Extract internal and external links
        domain = urlparse(url).netloc
        internal_links = []
        external_links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            anchor_text = a_tag.get_text(strip=True)
            
            # Get context (surrounding text)
            parent = a_tag.find_parent(['p', 'li', 'div', 'td'])
            context = parent.get_text(strip=True)[:200] if parent else ''
            
            # Check if dofollow
            is_dofollow = 'nofollow' not in a_tag.get('rel', [])
            
            # Resolve relative URLs
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            
            if parsed.netloc == domain or parsed.netloc == '':
                if not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                    internal_links.append({
                        'target_url': full_url,
                        'anchor_text': anchor_text,
                        'context': context,
                        'is_dofollow': is_dofollow
                    })
            elif parsed.scheme in ['http', 'https']:
                external_links.append({
                    'target_url': full_url,
                    'anchor_text': anchor_text
                })
        
        return {
            'meta_title': meta_title,
            'meta_description': meta_description,
            'h1': h1,
            'body_content': body_content,
            'word_count': word_count,
            'internal_links': internal_links,
            'external_links': external_links
        }
    
    async def crawl_urls(self, urls, progress_callback=None):
        """Crawl multiple URLs"""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=5)
        
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            tasks = [self.fetch_page(session, url) for url in urls]
            results = []
            
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, len(urls))
            
            return results

# Embedding and clustering functions
def generate_embeddings(texts, batch_size=32):
    """Generate embeddings for texts"""
    model = get_sentence_transformer()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def cluster_pages(embeddings, method='hdbscan', n_clusters=None, min_cluster_size=5):
    """Cluster pages based on embeddings"""
    if method == 'hdbscan':
        hdbscan_module = get_hdbscan()
        clusterer = hdbscan_module.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)
    else:  # kmeans
        _, KMeans = get_sklearn()
        if n_clusters is None:
            n_clusters = min(20, len(embeddings) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
    
    return labels

def reduce_dimensions(embeddings, method='umap'):
    """Reduce embeddings to 2D for visualization"""
    if method == 'umap':
        reducer = get_umap()
        reduced = reducer.fit_transform(embeddings)
    else:  # tsne
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced = tsne.fit_transform(embeddings)
    
    return reduced

def calculate_similarity(embeddings):
    """Calculate cosine similarity matrix"""
    cosine_similarity, _ = get_sklearn()
    return cosine_similarity(embeddings)

def calculate_pagerank(internal_links_df, damping=0.85, max_iter=100):
    """Calculate PageRank scores"""
    # Build adjacency matrix
    urls = set(internal_links_df['source_url'].unique()) | set(internal_links_df['target_url'].unique())
    url_to_idx = {url: i for i, url in enumerate(urls)}
    n = len(urls)
    
    if n == 0:
        return {}
    
    # Initialize
    pr = np.ones(n) / n
    out_degree = np.zeros(n)
    
    for _, row in internal_links_df.iterrows():
        src_idx = url_to_idx.get(row['source_url'])
        if src_idx is not None:
            out_degree[src_idx] += 1
    
    # Iterate
    for _ in range(max_iter):
        new_pr = np.ones(n) * (1 - damping) / n
        for _, row in internal_links_df.iterrows():
            src_idx = url_to_idx.get(row['source_url'])
            tgt_idx = url_to_idx.get(row['target_url'])
            if src_idx is not None and tgt_idx is not None and out_degree[src_idx] > 0:
                new_pr[tgt_idx] += damping * pr[src_idx] / out_degree[src_idx]
        pr = new_pr
    
    return {url: pr[idx] for url, idx in url_to_idx.items()}

def generate_link_suggestions(pages_df, similarity_matrix, existing_links_set, top_k=10):
    """Generate internal link suggestions based on similarity"""
    suggestions = []
    
    for i, source_row in pages_df.iterrows():
        source_url = source_row['url']
        
        # Get top similar pages
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip self
        
        for j in top_indices:
            target_row = pages_df.iloc[j]
            target_url = target_row['url']
            
            # Skip if link already exists
            if (source_url, target_url) in existing_links_set:
                continue
            
            similarity = similarities[j]
            if similarity < 0.3:  # Minimum threshold
                continue
            
            # Generate anchor text suggestion
            anchor_candidates = [
                target_row.get('h1', ''),
                target_row.get('meta_title', ''),
            ]
            suggested_anchor = next((a for a in anchor_candidates if a), target_url.split('/')[-1])
            
            # Determine priority
            if similarity > 0.7:
                priority = 'high'
            elif similarity > 0.5:
                priority = 'medium'
            else:
                priority = 'low'
            
            suggestions.append({
                'source_url': source_url,
                'target_url': target_url,
                'suggested_anchor': suggested_anchor[:100],
                'relevance_score': float(similarity),
                'action': 'add',
                'priority': priority,
                'reason': f'Semantic similarity: {similarity:.2f}'
            })
    
    return suggestions

def find_orphan_pages(pages_df, internal_links_df):
    """Find pages with no incoming internal links"""
    linked_urls = set(internal_links_df['target_url'].unique())
    all_urls = set(pages_df['url'].unique())
    orphans = all_urls - linked_urls
    return list(orphans)

def find_cannibalization(pages_df, similarity_matrix, threshold=0.85):
    """Find potentially cannibalizing pages"""
    cannibalization = []
    n = len(pages_df)
    
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] > threshold:
                cannibalization.append({
                    'url1': pages_df.iloc[i]['url'],
                    'url2': pages_df.iloc[j]['url'],
                    'title1': pages_df.iloc[i].get('meta_title', ''),
                    'title2': pages_df.iloc[j].get('meta_title', ''),
                    'similarity': similarity_matrix[i][j]
                })
    
    return cannibalization

# Streamlit UI
def main():
    # Initialize database
    conn = init_db()
    
    # Sidebar
    st.sidebar.markdown("## 🔗 Internal Link Builder Pro")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🕷️ Crawl URLs", "📊 Page Analysis", "🧩 Clusters & Embeddings", 
         "💡 Link Suggestions", "🔍 Orphan Pages", "⚠️ Cannibalization", "📥 Import/Export"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Crawler Settings")
    
    user_agent = st.sidebar.selectbox(
        "User Agent",
        list(USER_AGENTS.keys()),
        index=0,
        help="Googlebot is recommended to bypass some blocks"
    )
    
    if user_agent == 'custom':
        custom_ua = st.sidebar.text_input("Custom User Agent")
        USER_AGENTS['custom'] = custom_ua
    
    max_concurrent = st.sidebar.slider("Concurrent Requests", 1, 50, 10)
    request_delay = st.sidebar.slider("Delay (seconds)", 0.0, 5.0, 0.5, 0.1)
    timeout = st.sidebar.slider("Timeout (seconds)", 5, 120, 30)
    
    # Main content based on page
    if page == "🏠 Dashboard":
        show_dashboard(conn)
    elif page == "🕷️ Crawl URLs":
        show_crawl_page(conn, user_agent, max_concurrent, request_delay, timeout)
    elif page == "📊 Page Analysis":
        show_page_analysis(conn)
    elif page == "🧩 Clusters & Embeddings":
        show_clusters_page(conn)
    elif page == "💡 Link Suggestions":
        show_suggestions_page(conn)
    elif page == "🔍 Orphan Pages":
        show_orphan_pages(conn)
    elif page == "⚠️ Cannibalization":
        show_cannibalization_page(conn)
    elif page == "📥 Import/Export":
        show_import_export(conn)

def show_dashboard(conn):
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Overview of your internal linking analysis")
    
    # Get stats
    pages_count = pd.read_sql("SELECT COUNT(*) as count FROM pages WHERE crawl_status='success'", conn).iloc[0]['count']
    links_count = pd.read_sql("SELECT COUNT(*) as count FROM internal_links", conn).iloc[0]['count']
    suggestions_count = pd.read_sql("SELECT COUNT(*) as count FROM suggestions WHERE status='pending'", conn).iloc[0]['count']
    clusters_count = pd.read_sql("SELECT COUNT(DISTINCT cluster_id) as count FROM pages WHERE cluster_id IS NOT NULL", conn).iloc[0]['count']
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Pages Crawled", f"{pages_count:,}")
    with col2:
        st.metric("🔗 Internal Links", f"{links_count:,}")
    with col3:
        st.metric("💡 Pending Suggestions", f"{suggestions_count:,}")
    with col4:
        st.metric("🧩 Topic Clusters", clusters_count)
    
    st.markdown("---")
    
    if pages_count > 0:
        # Recent crawls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Crawl Status Distribution")
            status_df = pd.read_sql("""
                SELECT crawl_status, COUNT(*) as count 
                FROM pages 
                GROUP BY crawl_status
            """, conn)
            fig = px.pie(status_df, values='count', names='crawl_status', 
                        color_discrete_sequence=px.colors.sequential.Purples)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Pages by Cluster")
            cluster_df = pd.read_sql("""
                SELECT 
                    COALESCE(cluster_id, -1) as cluster_id,
                    COUNT(*) as count 
                FROM pages 
                WHERE crawl_status='success'
                GROUP BY cluster_id
                ORDER BY count DESC
                LIMIT 10
            """, conn)
            cluster_df['cluster_name'] = cluster_df['cluster_id'].apply(
                lambda x: f"Cluster {x}" if x >= 0 else "Unclustered"
            )
            fig = px.bar(cluster_df, x='cluster_name', y='count',
                        color='count', color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top pages by internal links
        st.subheader("🔝 Top Pages by Internal Links Received")
        top_pages = pd.read_sql("""
            SELECT 
                il.target_url as url,
                p.meta_title as title,
                COUNT(*) as incoming_links
            FROM internal_links il
            LEFT JOIN pages p ON il.target_url = p.url
            GROUP BY il.target_url
            ORDER BY incoming_links DESC
            LIMIT 10
        """, conn)
        st.dataframe(top_pages, use_container_width=True)
    else:
        st.info("👋 Welcome! Start by crawling some URLs in the 'Crawl URLs' section.")

def show_crawl_page(conn, user_agent, max_concurrent, request_delay, timeout):
    st.markdown('<h1 class="main-header">🕷️ Crawl URLs</h1>', unsafe_allow_html=True)
    st.markdown("Add URLs to crawl and analyze for internal linking opportunities")
    
    tab1, tab2 = st.tabs(["📝 Input URLs", "📂 Upload File"])
    
    with tab1:
        urls_text = st.text_area(
            "Enter URLs (one per line)",
            height=200,
            placeholder="https://example.com/page-1\nhttps://example.com/page-2\nhttps://example.com/page-3"
        )
    
    with tab2:
        uploaded_file = st.file_uploader("Upload CSV or TXT file with URLs", type=['csv', 'txt'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                url_column = st.selectbox("Select URL column", df.columns)
                urls_from_file = df[url_column].dropna().tolist()
            else:
                content = uploaded_file.read().decode('utf-8')
                urls_from_file = [u.strip() for u in content.split('\n') if u.strip()]
            st.success(f"Found {len(urls_from_file)} URLs in file")
            urls_text = '\n'.join(urls_from_file)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        start_crawl = st.button("🚀 Start Crawling", type="primary", use_container_width=True)
    with col2:
        clear_db = st.button("🗑️ Clear Database", use_container_width=True)
    
    if clear_db:
        c = conn.cursor()
        c.execute("DELETE FROM pages")
        c.execute("DELETE FROM internal_links")
        c.execute("DELETE FROM suggestions")
        c.execute("DELETE FROM clusters")
        conn.commit()
        st.success("Database cleared!")
        st.rerun()
    
    if start_crawl and urls_text:
        urls = [u.strip() for u in urls_text.split('\n') if u.strip().startswith('http')]
        
        if not urls:
            st.error("No valid URLs found. URLs must start with http:// or https://")
            return
        
        st.info(f"🔍 Starting crawl of {len(urls):,} URLs with {user_agent} user agent...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create crawler
        crawler = AsyncCrawler(
            user_agent=user_agent,
            max_concurrent=max_concurrent,
            timeout=timeout,
            delay=request_delay
        )
        
        # Run async crawl
        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Crawling: {current}/{total} URLs ({current/total*100:.1f}%)")
        
        # Execute crawl
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(crawler.crawl_urls(urls, update_progress))
        finally:
            loop.close()
        
        # Process results
        status_text.text("Processing crawled pages...")
        
        success_count = 0
        error_count = 0
        
        c = conn.cursor()
        
        for result in results:
            url = result['url']
            domain = urlparse(url).netloc
            
            if result['status'] == 'success':
                try:
                    parsed = crawler.parse_page(result['html'], url)
                    
                    # Insert page
                    c.execute("""
                        INSERT OR REPLACE INTO pages 
                        (url, domain, meta_title, meta_description, h1, body_content, 
                         word_count, internal_links_count, external_links_count, 
                         crawl_status, crawled_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        url, domain, parsed['meta_title'], parsed['meta_description'],
                        parsed['h1'], parsed['body_content'], parsed['word_count'],
                        len(parsed['internal_links']), len(parsed['external_links']),
                        'success', datetime.now()
                    ))
                    
                    # Insert internal links
                    for link in parsed['internal_links']:
                        try:
                            c.execute("""
                                INSERT OR IGNORE INTO internal_links 
                                (source_url, target_url, anchor_text, context, is_dofollow)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                url, link['target_url'], link['anchor_text'],
                                link['context'], link['is_dofollow']
                            ))
                        except:
                            pass
                    
                    success_count += 1
                except Exception as e:
                    c.execute("""
                        INSERT OR REPLACE INTO pages 
                        (url, domain, crawl_status, crawl_error, crawled_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (url, domain, 'error', str(e), datetime.now()))
                    error_count += 1
            else:
                c.execute("""
                    INSERT OR REPLACE INTO pages 
                    (url, domain, crawl_status, crawl_error, crawled_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (url, domain, 'error', result.get('error', 'Unknown'), datetime.now()))
                error_count += 1
        
        conn.commit()
        
        progress_bar.progress(1.0)
        status_text.text("Crawl complete!")
        
        st.success(f"""
        ✅ Crawl completed!
        - Successfully crawled: {success_count:,} pages
        - Errors: {error_count:,} pages
        """)
        
        # Show error summary if any
        if error_count > 0:
            with st.expander("View Errors"):
                errors_df = pd.read_sql("""
                    SELECT url, crawl_error 
                    FROM pages 
                    WHERE crawl_status='error'
                    ORDER BY crawled_at DESC
                    LIMIT 100
                """, conn)
                st.dataframe(errors_df, use_container_width=True)

def show_page_analysis(conn):
    st.markdown('<h1 class="main-header">📊 Page Analysis</h1>', unsafe_allow_html=True)
    
    pages_df = pd.read_sql("""
        SELECT 
            p.*,
            (SELECT COUNT(*) FROM internal_links WHERE source_url = p.url) as outgoing_links,
            (SELECT COUNT(*) FROM internal_links WHERE target_url = p.url) as incoming_links
        FROM pages p
        WHERE crawl_status = 'success'
        ORDER BY crawled_at DESC
    """, conn)
    
    if len(pages_df) == 0:
        st.info("No pages crawled yet. Go to 'Crawl URLs' to get started.")
        return
    
    st.markdown(f"**{len(pages_df):,} pages** successfully crawled")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        domain_filter = st.selectbox("Filter by Domain", ['All'] + pages_df['domain'].unique().tolist())
    with col2:
        min_words = st.number_input("Min Word Count", 0, 10000, 0)
    with col3:
        sort_by = st.selectbox("Sort By", ['crawled_at', 'word_count', 'incoming_links', 'outgoing_links'])
    
    # Apply filters
    filtered_df = pages_df.copy()
    if domain_filter != 'All':
        filtered_df = filtered_df[filtered_df['domain'] == domain_filter]
    filtered_df = filtered_df[filtered_df['word_count'] >= min_words]
    filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    # Display table
    display_cols = ['url', 'meta_title', 'h1', 'word_count', 'incoming_links', 'outgoing_links', 'cluster_id']
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=400
    )
    
    # Page detail view
    st.markdown("---")
    st.subheader("🔍 Page Details")
    
    selected_url = st.selectbox("Select a page to view details", filtered_df['url'].tolist())
    
    if selected_url:
        page_data = filtered_df[filtered_df['url'] == selected_url].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Meta Title:**")
            st.write(page_data['meta_title'] or 'N/A')
            
            st.markdown("**H1:**")
            st.write(page_data['h1'] or 'N/A')
            
            st.markdown("**Meta Description:**")
            st.write(page_data['meta_description'] or 'N/A')
        
        with col2:
            st.metric("Word Count", page_data['word_count'])
            st.metric("Incoming Links", page_data['incoming_links'])
            st.metric("Outgoing Links", page_data['outgoing_links'])
        
        # Show internal links from this page
        with st.expander("View Outgoing Internal Links"):
            links_df = pd.read_sql("""
                SELECT target_url, anchor_text, context
                FROM internal_links
                WHERE source_url = ?
            """, conn, params=[selected_url])
            st.dataframe(links_df, use_container_width=True)
        
        # Show internal links to this page
        with st.expander("View Incoming Internal Links"):
            incoming_df = pd.read_sql("""
                SELECT source_url, anchor_text, context
                FROM internal_links
                WHERE target_url = ?
            """, conn, params=[selected_url])
            st.dataframe(incoming_df, use_container_width=True)

def show_clusters_page(conn):
    st.markdown('<h1 class="main-header">🧩 Clusters & Embeddings</h1>', unsafe_allow_html=True)
    st.markdown("Visualize semantic relationships and topic clusters")
    
    pages_df = pd.read_sql("""
        SELECT url, meta_title, h1, body_content, cluster_id
        FROM pages
        WHERE crawl_status = 'success' AND body_content IS NOT NULL
    """, conn)
    
    if len(pages_df) < 5:
        st.warning("Need at least 5 successfully crawled pages with content to perform clustering.")
        return
    
    st.info(f"📄 {len(pages_df):,} pages available for clustering")
    
    # Clustering settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clustering_method = st.selectbox("Clustering Method", ['hdbscan', 'kmeans'])
    
    with col2:
        if clustering_method == 'kmeans':
            n_clusters = st.slider("Number of Clusters", 2, 50, min(10, len(pages_df)//5))
        else:
            min_cluster_size = st.slider("Min Cluster Size", 2, 20, 5)
    
    with col3:
        dim_reduction = st.selectbox("Dimension Reduction", ['umap', 'tsne'])
    
    if st.button("🔮 Generate Embeddings & Clusters", type="primary"):
        with st.spinner("Generating embeddings... This may take a few minutes for large datasets."):
            # Prepare text for embedding
            texts = []
            for _, row in pages_df.iterrows():
                text = f"{row['meta_title'] or ''} {row['h1'] or ''} {row['body_content'] or ''}"
                texts.append(text[:2000])  # Limit text length
            
            # Generate embeddings
            embeddings = generate_embeddings(texts)
            
            # Save embeddings to database
            c = conn.cursor()
            for i, row in pages_df.iterrows():
                embedding_bytes = embeddings[i].tobytes()
                c.execute("UPDATE pages SET embedding = ? WHERE url = ?", (embedding_bytes, row['url']))
            conn.commit()
            
            st.success("✅ Embeddings generated!")
        
        with st.spinner("Clustering pages..."):
            # Cluster
            if clustering_method == 'kmeans':
                labels = cluster_pages(embeddings, method='kmeans', n_clusters=n_clusters)
            else:
                labels = cluster_pages(embeddings, method='hdbscan', min_cluster_size=min_cluster_size)
            
            # Save cluster labels
            c = conn.cursor()
            for i, row in pages_df.iterrows():
                c.execute("UPDATE pages SET cluster_id = ? WHERE url = ?", (int(labels[i]), row['url']))
            conn.commit()
            
            st.success("✅ Clustering complete!")
        
        with st.spinner("Reducing dimensions for visualization..."):
            # Dimension reduction
            reduced = reduce_dimensions(embeddings, method=dim_reduction)
            
            # Create visualization dataframe
            viz_df = pages_df.copy()
            viz_df['x'] = reduced[:, 0]
            viz_df['y'] = reduced[:, 1]
            viz_df['cluster'] = labels
            viz_df['cluster_label'] = viz_df['cluster'].apply(lambda x: f"Cluster {x}" if x >= 0 else "Noise")
            
            st.session_state['viz_df'] = viz_df
            st.session_state['embeddings'] = embeddings
            st.success("✅ Visualization ready!")
    
    # Display visualization if available
    if 'viz_df' in st.session_state:
        viz_df = st.session_state['viz_df']
        
        st.subheader("📊 Semantic Cluster Visualization")
        
        # Color by cluster
        fig = px.scatter(
            viz_df,
            x='x', y='y',
            color='cluster_label',
            hover_data=['url', 'meta_title', 'h1'],
            title=f"Pages clustered by semantic similarity ({dim_reduction.upper()})",
            height=600
        )
        fig.update_layout(
            xaxis_title=f"{dim_reduction.upper()} Dimension 1",
            yaxis_title=f"{dim_reduction.upper()} Dimension 2"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.subheader("📈 Cluster Statistics")
        
        cluster_stats = viz_df.groupby('cluster_label').agg({
            'url': 'count',
            'meta_title': lambda x: x.iloc[0] if len(x) > 0 else ''
        }).rename(columns={'url': 'Page Count', 'meta_title': 'Example Page'})
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Similarity matrix visualization
        if st.checkbox("Show Similarity Heatmap (for small datasets)"):
            if len(viz_df) <= 100:
                embeddings = st.session_state['embeddings']
                sim_matrix = calculate_similarity(embeddings)
                
                fig = px.imshow(
                    sim_matrix,
                    labels=dict(color="Similarity"),
                    title="Page Similarity Matrix",
                    color_continuous_scale="Purples"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Similarity heatmap is only available for datasets with 100 or fewer pages.")

def show_suggestions_page(conn):
    st.markdown('<h1 class="main-header">💡 Link Suggestions</h1>', unsafe_allow_html=True)
    
    pages_df = pd.read_sql("""
        SELECT url, meta_title, h1, body_content
        FROM pages
        WHERE crawl_status = 'success'
    """, conn)
    
    if len(pages_df) < 2:
        st.warning("Need at least 2 crawled pages to generate suggestions.")
        return
    
    # Check if embeddings exist
    embeddings_exist = pd.read_sql("""
        SELECT COUNT(*) as count FROM pages WHERE embedding IS NOT NULL
    """, conn).iloc[0]['count']
    
    if embeddings_exist < len(pages_df):
        st.warning("⚠️ Not all pages have embeddings. Go to 'Clusters & Embeddings' first to generate them.")
        if st.button("Quick Generate Embeddings"):
            with st.spinner("Generating embeddings..."):
                texts = []
                for _, row in pages_df.iterrows():
                    text = f"{row['meta_title'] or ''} {row['h1'] or ''} {row['body_content'] or ''}"
                    texts.append(text[:2000])
                
                embeddings = generate_embeddings(texts)
                
                c = conn.cursor()
                for i, row in pages_df.iterrows():
                    embedding_bytes = embeddings[i].tobytes()
                    c.execute("UPDATE pages SET embedding = ? WHERE url = ?", (embedding_bytes, row['url']))
                conn.commit()
                
                st.success("✅ Embeddings generated!")
                st.rerun()
    
    # Generate suggestions button
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.number_input("Suggestions per page", 1, 20, 5)
    
    if st.button("🔮 Generate Link Suggestions", type="primary"):
        with st.spinner("Analyzing pages and generating suggestions..."):
            # Load embeddings
            pages_with_emb = pd.read_sql("""
                SELECT url, meta_title, h1, embedding
                FROM pages
                WHERE crawl_status = 'success' AND embedding IS NOT NULL
            """, conn)
            
            # Convert embeddings
            embeddings = np.array([
                np.frombuffer(row['embedding'], dtype=np.float32)
                for _, row in pages_with_emb.iterrows()
            ])
            
            # Calculate similarity
            sim_matrix = calculate_similarity(embeddings)
            
            # Get existing links
            existing_links = pd.read_sql("SELECT source_url, target_url FROM internal_links", conn)
            existing_links_set = set(zip(existing_links['source_url'], existing_links['target_url']))
            
            # Generate suggestions
            suggestions = generate_link_suggestions(pages_with_emb, sim_matrix, existing_links_set, top_k=top_k)
            
            # Save to database
            c = conn.cursor()
            c.execute("DELETE FROM suggestions")  # Clear old suggestions
            
            for s in suggestions:
                c.execute("""
                    INSERT INTO suggestions 
                    (source_url, target_url, suggested_anchor, relevance_score, action, priority, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    s['source_url'], s['target_url'], s['suggested_anchor'],
                    s['relevance_score'], s['action'], s['priority'], s['reason']
                ))
            
            conn.commit()
            st.success(f"✅ Generated {len(suggestions):,} link suggestions!")
    
    # Display suggestions
    suggestions_df = pd.read_sql("""
        SELECT * FROM suggestions
        ORDER BY relevance_score DESC
    """, conn)
    
    if len(suggestions_df) > 0:
        st.markdown("---")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            high_priority = len(suggestions_df[suggestions_df['priority'] == 'high'])
            st.metric("🔴 High Priority", high_priority)
        with col2:
            medium_priority = len(suggestions_df[suggestions_df['priority'] == 'medium'])
            st.metric("🟡 Medium Priority", medium_priority)
        with col3:
            low_priority = len(suggestions_df[suggestions_df['priority'] == 'low'])
            st.metric("🟢 Low Priority", low_priority)
        
        # Filter
        priority_filter = st.multiselect(
            "Filter by Priority",
            ['high', 'medium', 'low'],
            default=['high', 'medium']
        )
        
        filtered_suggestions = suggestions_df[suggestions_df['priority'].isin(priority_filter)]
        
        # Display
        st.dataframe(
            filtered_suggestions[['source_url', 'target_url', 'suggested_anchor', 'relevance_score', 'priority', 'reason']],
            use_container_width=True,
            height=400
        )
        
        # Export suggestions
        csv = filtered_suggestions.to_csv(index=False)
        st.download_button(
            "📥 Download Suggestions CSV",
            csv,
            "link_suggestions.csv",
            "text/csv"
        )
    else:
        st.info("No suggestions yet. Click 'Generate Link Suggestions' to analyze your pages.")

def show_orphan_pages(conn):
    st.markdown('<h1 class="main-header">🔍 Orphan Pages</h1>', unsafe_allow_html=True)
    st.markdown("Pages with no incoming internal links")
    
    # Find orphan pages
    orphans_df = pd.read_sql("""
        SELECT p.url, p.meta_title, p.h1, p.word_count,
               (SELECT COUNT(*) FROM internal_links WHERE source_url = p.url) as outgoing_links
        FROM pages p
        WHERE p.crawl_status = 'success'
        AND p.url NOT IN (SELECT DISTINCT target_url FROM internal_links)
        ORDER BY p.word_count DESC
    """, conn)
    
    if len(orphans_df) == 0:
        st.success("🎉 No orphan pages found! All pages have incoming internal links.")
        return
    
    st.warning(f"⚠️ Found {len(orphans_df):,} orphan pages without incoming internal links")
    
    st.dataframe(orphans_df, use_container_width=True, height=400)
    
    # Download
    csv = orphans_df.to_csv(index=False)
    st.download_button(
        "📥 Download Orphan Pages CSV",
        csv,
        "orphan_pages.csv",
        "text/csv"
    )

def show_cannibalization_page(conn):
    st.markdown('<h1 class="main-header">⚠️ Content Cannibalization</h1>', unsafe_allow_html=True)
    st.markdown("Pages that may be competing for the same keywords")
    
    # Check if embeddings exist
    embeddings_count = pd.read_sql("""
        SELECT COUNT(*) as count FROM pages WHERE embedding IS NOT NULL
    """, conn).iloc[0]['count']
    
    if embeddings_count < 2:
        st.warning("Need embeddings to detect cannibalization. Go to 'Clusters & Embeddings' first.")
        return
    
    threshold = st.slider("Similarity Threshold", 0.5, 0.99, 0.85, 0.01,
                         help="Pages with similarity above this threshold may be cannibalizing")
    
    if st.button("🔍 Detect Cannibalization", type="primary"):
        with st.spinner("Analyzing content similarity..."):
            # Load pages with embeddings
            pages_df = pd.read_sql("""
                SELECT url, meta_title, h1, embedding
                FROM pages
                WHERE crawl_status = 'success' AND embedding IS NOT NULL
            """, conn)
            
            # Convert embeddings
            embeddings = np.array([
                np.frombuffer(row['embedding'], dtype=np.float32)
                for _, row in pages_df.iterrows()
            ])
            
            # Calculate similarity
            sim_matrix = calculate_similarity(embeddings)
            
            # Find cannibalization
            cannibalization = find_cannibalization(pages_df, sim_matrix, threshold)
            
            st.session_state['cannibalization'] = cannibalization
    
    if 'cannibalization' in st.session_state:
        cannibalization = st.session_state['cannibalization']
        
        if len(cannibalization) == 0:
            st.success(f"🎉 No cannibalization detected at {threshold:.0%} similarity threshold!")
        else:
            st.warning(f"⚠️ Found {len(cannibalization):,} potentially cannibalizing page pairs")
            
            cann_df = pd.DataFrame(cannibalization)
            cann_df['similarity'] = cann_df['similarity'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(cann_df, use_container_width=True)
            
            csv = cann_df.to_csv(index=False)
            st.download_button(
                "📥 Download Cannibalization Report",
                csv,
                "cannibalization_report.csv",
                "text/csv"
            )

def show_import_export(conn):
    st.markdown('<h1 class="main-header">📥 Import/Export</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Export Data")
        
        export_format = st.selectbox("Export Format", ['CSV', 'JSON'])
        
        if st.button("Export All Data"):
            # Export pages
            pages_df = pd.read_sql("SELECT * FROM pages", conn)
            links_df = pd.read_sql("SELECT * FROM internal_links", conn)
            suggestions_df = pd.read_sql("SELECT * FROM suggestions", conn)
            
            if export_format == 'CSV':
                # Create zip with multiple CSVs
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr('pages.csv', pages_df.to_csv(index=False))
                    zf.writestr('internal_links.csv', links_df.to_csv(index=False))
                    zf.writestr('suggestions.csv', suggestions_df.to_csv(index=False))
                
                st.download_button(
                    "📥 Download ZIP",
                    zip_buffer.getvalue(),
                    "internal_links_export.zip",
                    "application/zip"
                )
            else:
                export_data = {
                    'pages': pages_df.to_dict(orient='records'),
                    'internal_links': links_df.to_dict(orient='records'),
                    'suggestions': suggestions_df.to_dict(orient='records'),
                    'exported_at': datetime.now().isoformat()
                }
                
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(export_data, indent=2, default=str),
                    "internal_links_export.json",
                    "application/json"
                )
    
    with col2:
        st.subheader("📥 Import Data")
        
        uploaded_file = st.file_uploader("Upload JSON export file", type=['json'])
        
        if uploaded_file:
            try:
                data = json.loads(uploaded_file.read())
                
                st.info(f"""
                Found:
                - {len(data.get('pages', []))} pages
                - {len(data.get('internal_links', []))} internal links
                - {len(data.get('suggestions', []))} suggestions
                """)
                
                if st.button("Import Data", type="primary"):
                    c = conn.cursor()
                    
                    # Import pages
                    for page in data.get('pages', []):
                        c.execute("""
                            INSERT OR REPLACE INTO pages 
                            (url, domain, meta_title, meta_description, h1, body_content,
                             word_count, internal_links_count, external_links_count,
                             crawl_status, crawl_error, crawled_at, cluster_id, page_rank)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            page.get('url'), page.get('domain'), page.get('meta_title'),
                            page.get('meta_description'), page.get('h1'), page.get('body_content'),
                            page.get('word_count'), page.get('internal_links_count'),
                            page.get('external_links_count'), page.get('crawl_status'),
                            page.get('crawl_error'), page.get('crawled_at'),
                            page.get('cluster_id'), page.get('page_rank', 0)
                        ))
                    
                    # Import links
                    for link in data.get('internal_links', []):
                        c.execute("""
                            INSERT OR IGNORE INTO internal_links 
                            (source_url, target_url, anchor_text, context, is_dofollow)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            link.get('source_url'), link.get('target_url'),
                            link.get('anchor_text'), link.get('context'),
                            link.get('is_dofollow', 1)
                        ))
                    
                    conn.commit()
                    st.success("✅ Data imported successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error importing file: {e}")

if __name__ == "__main__":
    main()
