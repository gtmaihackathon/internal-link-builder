"""
Internal Link Builder - AI-Powered SEO Tool v2.0
Fixed version with proper AI Settings
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import json
import re
import os

def safe_score_format(value, decimals=3):
    """Safely format a score value, returning 'N/A' if invalid."""
    try:
        if value is None or value == '' or value == 'None':
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

def safe_get(dictionary, key, default=''):
    """Safely get a value from a dictionary, handling None values."""
    value = dictionary.get(key, default)
    return value if value is not None else default
        
# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Internal Link Builder - AI Powered",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #FF4B4B; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .content-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .link-group { background-color: #e7f3ff; border-left: 4px solid #0066cc; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }
    .similarity-high { color: #28a745; font-weight: bold; }
    .similarity-medium { color: #ffc107; font-weight: bold; }
    .similarity-low { color: #dc3545; font-weight: bold; }
    .body-content-preview { background-color: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; font-size: 0.9rem; line-height: 1.6; max-height: 500px; overflow-y: auto; white-space: pre-wrap; }
    .api-card { background: #f8f9fa; border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0; border: 1px solid #dee2e6; }
    .api-active { border-left: 4px solid #28a745; }
    .api-inactive { border-left: 4px solid #6c757d; }
</style>
""", unsafe_allow_html=True)


# ============ DATABASE FUNCTIONS ============

def init_database():
    """Initialize SQLite database"""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/internal_links.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Pages table
    cursor.execute('''CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        domain TEXT,
        meta_title TEXT,
        meta_description TEXT,
        h1 TEXT,
        h2_tags TEXT,
        body_content TEXT,
        word_count INTEGER DEFAULT 0,
        internal_links_count INTEGER DEFAULT 0,
        external_links_count INTEGER DEFAULT 0,
        crawl_status TEXT DEFAULT 'pending',
        crawl_error TEXT,
        http_status INTEGER,
        crawled_at TIMESTAMP,
        embedding BLOB,
        cluster_id INTEGER,
        cluster_name TEXT,
        page_rank REAL DEFAULT 0.0
    )''')
    
    # Internal links table
    cursor.execute('''CREATE TABLE IF NOT EXISTS internal_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT NOT NULL,
        target_url TEXT NOT NULL,
        anchor_text TEXT,
        context TEXT,
        is_dofollow INTEGER DEFAULT 1,
        similarity_score REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(source_url, target_url, anchor_text)
    )''')
    
    # Suggestions table
    cursor.execute('''CREATE TABLE IF NOT EXISTS suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT NOT NULL,
        target_url TEXT NOT NULL,
        suggested_anchor TEXT,
        suggested_context TEXT,
        relevance_score REAL,
        priority TEXT DEFAULT 'medium',
        reason TEXT,
        ai_explanation TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(source_url, target_url)
    )''')
    
    # API settings table
    cursor.execute('''CREATE TABLE IF NOT EXISTS api_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT UNIQUE NOT NULL,
        api_key TEXT,
        model TEXT,
        is_active INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    return conn


# ============ AI PROVIDER CLASS ============

class AIProvider:
    """Handle AI API operations"""
    
    def __init__(self, conn):
        self.conn = conn
    
    def get_active_provider(self) -> Optional[Dict]:
        """Get currently active AI provider"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM api_settings WHERE is_active = 1 LIMIT 1')
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_all_providers(self) -> List[Dict]:
        """Get all saved providers"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM api_settings ORDER BY is_active DESC')
        return [dict(row) for row in cursor.fetchall()]
    
    def save_api_key(self, provider: str, api_key: str, model: str, set_active: bool = True):
        """Save or update API key"""
        cursor = self.conn.cursor()
        
        # Deactivate all if setting this as active
        if set_active:
            cursor.execute('UPDATE api_settings SET is_active = 0')
        
        # Insert or update
        cursor.execute('''
            INSERT INTO api_settings (provider, api_key, model, is_active, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                api_key = excluded.api_key,
                model = excluded.model,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
        ''', (provider, api_key, model, 1 if set_active else 0, datetime.now()))
        
        self.conn.commit()
    
    def delete_provider(self, provider: str):
        """Delete a provider"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM api_settings WHERE provider = ?', (provider,))
        self.conn.commit()
    
    def set_active(self, provider: str):
        """Set a provider as active"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE api_settings SET is_active = 0')
        cursor.execute('UPDATE api_settings SET is_active = 1 WHERE provider = ?', (provider,))
        self.conn.commit()
    
    async def call_ai(self, prompt: str, provider_settings: Dict) -> str:
        """Call AI API"""
        provider = provider_settings['provider']
        api_key = provider_settings['api_key']
        model = provider_settings['model']
        
        try:
            if provider == 'openai':
                return await self._call_openai(prompt, api_key, model)
            elif provider == 'anthropic':
                return await self._call_anthropic(prompt, api_key, model)
            elif provider == 'google':
                return await self._call_google(prompt, api_key, model)
            else:
                return f"Unknown provider: {provider}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _call_openai(self, prompt: str, api_key: str, model: str) -> str:
        """Call OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except ImportError:
            return "Error: openai package not installed. Run: pip install openai"
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    async def _call_anthropic(self, prompt: str, api_key: str, model: str) -> str:
        """Call Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except ImportError:
            return "Error: anthropic package not installed. Run: pip install anthropic"
        except Exception as e:
            return f"Anthropic Error: {str(e)}"
    
    async def _call_google(self, prompt: str, api_key: str, model: str) -> str:
        """Call Google Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        except ImportError:
            return "Error: google-generativeai package not installed. Run: pip install google-generativeai"
        except Exception as e:
            return f"Google Error: {str(e)}"


# ============ CRAWLER FUNCTIONS ============

USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
}


def extract_body_content(soup: BeautifulSoup) -> str:
    """Extract clean body content"""
    # Remove unwanted tags
    for tag in soup.find_all(['script', 'style', 'noscript', 'iframe', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()
    
    # Find main content
    main_content = None
    for selector in ['main', 'article', '[role="main"]', '.content', '.post-content', '.entry-content', '#content', '.article-body']:
        if selector.startswith(('.', '#', '[')):
            main_content = soup.select_one(selector)
        else:
            main_content = soup.find(selector)
        if main_content:
            break
    
    if not main_content:
        main_content = soup.find('body')
    
    if main_content:
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'blockquote'])
        text_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 15:
                text_parts.append(text)
        return '\n\n'.join(text_parts)
    
    return ''


async def crawl_url(session: aiohttp.ClientSession, url: str, user_agent: str) -> Dict:
    """Crawl a single URL"""
    result = {
        'url': url,
        'status': 'pending',
        'meta_title': '',
        'meta_description': '',
        'h1': '',
        'h2_tags': [],
        'body_content': '',
        'word_count': 0,
        'internal_links': [],
        'external_links': [],
        'error': '',
        'http_status': 0
    }
    
    try:
        headers = {'User-Agent': USER_AGENTS.get(user_agent, USER_AGENTS['googlebot'])}
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with session.get(url, headers=headers, timeout=timeout, ssl=False) as response:
            result['http_status'] = response.status
            
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                domain = urlparse(url).netloc
                
                # Extract title
                title_tag = soup.find('title')
                result['meta_title'] = title_tag.get_text(strip=True) if title_tag else ''
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                result['meta_description'] = meta_desc.get('content', '') if meta_desc else ''
                
                # Extract H1
                h1_tag = soup.find('h1')
                result['h1'] = h1_tag.get_text(strip=True) if h1_tag else ''
                
                # Extract H2 tags
                result['h2_tags'] = [h2.get_text(strip=True) for h2 in soup.find_all('h2')[:10]]
                
                # Extract body content
                result['body_content'] = extract_body_content(soup)
                result['word_count'] = len(result['body_content'].split())
                
                # Extract links
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        continue
                    
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)
                    
                    if not parsed.scheme or not parsed.netloc:
                        continue
                    
                    anchor_text = a_tag.get_text(strip=True)
                    parent = a_tag.find_parent(['p', 'li', 'div', 'td'])
                    context = parent.get_text(strip=True)[:200] if parent else ''
                    
                    rel = a_tag.get('rel', [])
                    is_dofollow = 'nofollow' not in (rel if isinstance(rel, list) else [rel])
                    
                    link_data = {
                        'target_url': full_url,
                        'anchor_text': anchor_text,
                        'context': context,
                        'is_dofollow': is_dofollow
                    }
                    
                    if parsed.netloc == domain or parsed.netloc.endswith('.' + domain):
                        result['internal_links'].append(link_data)
                    else:
                        result['external_links'].append(link_data)
                
                result['status'] = 'success'
            else:
                result['status'] = 'error'
                result['error'] = f'HTTP {response.status}'
                
    except asyncio.TimeoutError:
        result['status'] = 'error'
        result['error'] = 'Timeout'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)[:200]
    
    return result


async def crawl_urls_async(urls: List[str], user_agent: str, max_concurrent: int, delay: float, progress_callback=None) -> List[Dict]:
    """Crawl multiple URLs"""
    connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=False)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_limit(url: str):
            async with semaphore:
                if delay > 0:
                    await asyncio.sleep(delay)
                return await crawl_url(session, url, user_agent)
        
        tasks = [crawl_with_limit(url) for url in urls]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(urls))
        
        return results


def run_crawler(urls: List[str], user_agent: str, max_concurrent: int, delay: float, progress_callback=None) -> List[Dict]:
    """Synchronous wrapper for crawler"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            crawl_urls_async(urls, user_agent, max_concurrent, delay, progress_callback)
        )
    finally:
        loop.close()


# ============ EMBEDDING FUNCTIONS ============

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for texts"""
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=True)


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity matrix"""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


def cluster_pages(embeddings: np.ndarray, method: str = 'hdbscan', **kwargs):
    """Cluster embeddings"""
    if method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=kwargs.get('min_cluster_size', 5),
            min_samples=kwargs.get('min_samples', None)
        )
        return clusterer.fit_predict(embeddings)
    else:
        from sklearn.cluster import KMeans
        n_clusters = kwargs.get('n_clusters', 10)
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)


def reduce_dimensions(embeddings: np.ndarray, method: str = 'umap') -> np.ndarray:
    """Reduce to 2D for visualization"""
    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
    
    return reducer.fit_transform(embeddings)


# ============ PAGE FUNCTIONS ============

def show_dashboard(conn):
    """Dashboard page"""
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Overview of your internal linking analysis</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    cursor.execute("SELECT COUNT(*) FROM pages WHERE crawl_status = 'success'")
    col1.metric("📄 Pages Crawled", cursor.fetchone()[0])
    
    cursor.execute("SELECT COUNT(*) FROM internal_links")
    col2.metric("🔗 Internal Links", cursor.fetchone()[0])
    
    cursor.execute("SELECT COUNT(DISTINCT cluster_id) FROM pages WHERE cluster_id IS NOT NULL AND cluster_id >= 0")
    col3.metric("🎯 Clusters", cursor.fetchone()[0])
    
    cursor.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'")
    col4.metric("💡 Opportunities", cursor.fetchone()[0])
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Pages by Cluster")
        cursor.execute("""
            SELECT COALESCE(cluster_name, 'Unclustered') as cluster, COUNT(*) as count 
            FROM pages WHERE crawl_status = 'success'
            GROUP BY cluster_id ORDER BY count DESC LIMIT 10
        """)
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['Cluster', 'Count'])
            fig = px.pie(df, values='Count', names='Cluster', hole=0.4)
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster data yet. Generate embeddings first.")
    
    with col2:
        st.subheader("🔗 Top Linked Pages")
        cursor.execute("""
            SELECT target_url, COUNT(*) as incoming
            FROM internal_links GROUP BY target_url
            ORDER BY incoming DESC LIMIT 10
        """)
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['URL', 'Incoming Links'])
            df['URL'] = df['URL'].apply(lambda x: x.split('/')[-1][:25] if x else 'home')
            fig = px.bar(df, x='Incoming Links', y='URL', orientation='h')
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No link data yet. Crawl URLs first.")
    
    # Orphan pages
    st.subheader("🚨 Orphan Pages (No Incoming Links)")
    cursor.execute("""
        SELECT url, meta_title, word_count 
        FROM pages WHERE crawl_status = 'success'
        AND url NOT IN (SELECT DISTINCT target_url FROM internal_links)
        LIMIT 10
    """)
    orphans = cursor.fetchall()
    if orphans:
        df = pd.DataFrame(orphans, columns=['URL', 'Title', 'Word Count'])
        st.dataframe(df, use_container_width=True)
    else:
        st.success("✅ No orphan pages found!")
    
    return None  # Explicit return


def show_crawl_page(conn):
    """Crawl URLs page"""
    st.markdown('<p class="main-header">🔍 Crawl URLs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Add and crawl URLs for analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        urls_input = st.text_area(
            "Enter URLs (one per line)",
            height=200,
            placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3"
        )
    
    with col2:
        st.markdown("**Settings**")
        user_agent = st.selectbox("User Agent", ['googlebot', 'bingbot', 'chrome'])
        max_concurrent = st.slider("Concurrent", 1, 50, 10)
        delay = st.slider("Delay (s)", 0.0, 5.0, 0.5, 0.1)
    
    # File upload
    uploaded = st.file_uploader("Or upload CSV/TXT file", type=['csv', 'txt'])
    if uploaded:
        content = uploaded.getvalue().decode('utf-8')
        if uploaded.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded)
                url_col = [c for c in df.columns if 'url' in c.lower()]
                if url_col:
                    urls_input = '\n'.join(df[url_col[0]].dropna().tolist())
                else:
                    urls_input = '\n'.join(df.iloc[:, 0].dropna().tolist())
            except:
                urls_input = content
        else:
            urls_input = content
    
    if st.button("🚀 Start Crawling", type="primary"):
        urls = [u.strip() for u in urls_input.strip().split('\n') if u.strip() and u.strip().startswith('http')]
        
        if not urls:
            st.error("Please enter at least one valid URL (starting with http)")
            return None
        
        st.info(f"Starting crawl of {len(urls)} URLs...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Crawling: {current}/{total}")
        
        # Run crawler
        results = run_crawler(urls, user_agent, max_concurrent, delay, update_progress)
        
        # Save results
        cursor = conn.cursor()
        success_count = 0
        error_count = 0
        
        for r in results:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO pages 
                    (url, domain, meta_title, meta_description, h1, h2_tags, body_content,
                     word_count, internal_links_count, external_links_count, crawl_status, 
                     crawl_error, http_status, crawled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    r['url'],
                    urlparse(r['url']).netloc,
                    r['meta_title'],
                    r['meta_description'],
                    r['h1'],
                    json.dumps(r['h2_tags']),
                    r['body_content'],
                    r['word_count'],
                    len(r['internal_links']),
                    len(r['external_links']),
                    r['status'],
                    r.get('error', ''),
                    r.get('http_status', 0),
                    datetime.now()
                ))
                
                # Save links
                for link in r['internal_links']:
                    try:
                        cursor.execute('''
                            INSERT OR IGNORE INTO internal_links 
                            (source_url, target_url, anchor_text, context, is_dofollow)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            r['url'],
                            link['target_url'],
                            link['anchor_text'],
                            link['context'],
                            1 if link['is_dofollow'] else 0
                        ))
                    except:
                        pass
                
                if r['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
        
        conn.commit()
        st.success(f"✅ Crawl complete! {success_count} succeeded, {error_count} failed")
        
        # Show results
        results_df = pd.DataFrame([{
            'URL': r['url'][:60],
            'Status': '✅' if r['status'] == 'success' else '❌',
            'Title': (r['meta_title'] or '')[:40],
            'Words': r['word_count'],
            'Links': len(r['internal_links'])
        } for r in results])
        st.dataframe(results_df, use_container_width=True)
    
    return None


def show_page_analysis(conn, ai_provider):
    """Page Analysis page with body content"""
    st.markdown('<p class="main-header">📄 Page Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed analysis with full content extraction</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    cursor.execute("SELECT url, meta_title, word_count FROM pages WHERE crawl_status = 'success' ORDER BY crawled_at DESC")
    pages = cursor.fetchall()
    
    if not pages:
        st.warning("No pages crawled yet. Go to 'Crawl URLs' first.")
        return None
    
    # Page selector
    page_urls = [p[0] for p in pages]
    selected_url = st.selectbox(
        "Select a page to analyze",
        page_urls,
        format_func=lambda x: f"{x[:80]}..." if len(x) > 80 else x
    )
    
    if selected_url:
        cursor.execute('SELECT * FROM pages WHERE url = ?', (selected_url,))
        row = cursor.fetchone()
        
        if row:
            page = dict(row)
            
            st.markdown(f"### 📄 {page['meta_title'] or 'Untitled Page'}")
            st.markdown(f"**URL:** [{selected_url}]({selected_url})")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📝 Content", "🔗 Links", "🤖 AI Analysis"])
            
            with tab1:
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Words", page['word_count'])
                c2.metric("Internal Links", page['internal_links_count'])
                c3.metric("External Links", page['external_links_count'])
                c4.metric("Cluster", page['cluster_name'] or 'N/A')
                
                st.divider()
                
                # Meta info
                st.subheader("🏷️ Meta Information")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Meta Title:**")
                    st.info(page['meta_title'] or 'Not found')
                    tlen = len(page['meta_title'] or '')
                    if tlen < 30 or tlen > 60:
                        st.warning(f"⚠️ Title length: {tlen} chars (aim for 50-60)")
                    else:
                        st.success(f"✅ Good length: {tlen} chars")
                
                with col2:
                    st.markdown("**Meta Description:**")
                    st.info(page['meta_description'] or 'Not found')
                    dlen = len(page['meta_description'] or '')
                    if dlen < 120 or dlen > 160:
                        st.warning(f"⚠️ Description length: {dlen} chars (aim for 150-160)")
                    else:
                        st.success(f"✅ Good length: {dlen} chars")
                
                st.markdown("**H1 Tag:**")
                st.info(page['h1'] or 'Not found')
                
                if page['h2_tags']:
                    st.markdown("**H2 Tags:**")
                    try:
                        h2_list = json.loads(page['h2_tags']) if isinstance(page['h2_tags'], str) else page['h2_tags']
                        for h2 in h2_list:
                            st.markdown(f"- {h2}")
                    except:
                        st.text(page['h2_tags'])
            
            with tab2:
                st.subheader("📝 Page Content")
                
                body_content = page['body_content'] or ''
                
                if body_content:
                    words = body_content.split()
                    sentences = [s for s in re.split(r'[.!?]+', body_content) if s.strip()]
                    paragraphs = [p for p in body_content.split('\n\n') if p.strip()]
                    
                    # Stats
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Words", len(words))
                    c2.metric("Sentences", len(sentences))
                    c3.metric("Paragraphs", len(paragraphs))
                    c4.metric("Avg Words/Sentence", round(len(words) / max(len(sentences), 1), 1))
                    
                    st.divider()
                    
                    # Display options
                    display_mode = st.radio(
                        "Display mode",
                        ["Full Content", "First 500 words", "First 200 words"],
                        horizontal=True
                    )
                    
                    if display_mode == "Full Content":
                        display_text = body_content
                    elif display_mode == "First 500 words":
                        display_text = ' '.join(words[:500])
                    else:
                        display_text = ' '.join(words[:200])
                    
                    st.markdown(f'<div class="body-content-preview">{display_text}</div>', unsafe_allow_html=True)
                    
                    # Download
                    st.download_button(
                        "📥 Download Content",
                        body_content,
                        file_name=f"content_{urlparse(selected_url).path.replace('/', '_')[:50]}.txt",
                        mime="text/plain"
                    )
                    
                    st.divider()
                    
                    # Content analysis
                    st.subheader("📊 Content Quality Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Readability Metrics:**")
                        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
                        st.markdown(f"- Average word length: {avg_word_len:.1f} chars")
                        long_sentences = sum(1 for s in sentences if len(s.split()) > 25)
                        st.markdown(f"- Long sentences (>25 words): {long_sentences}")
                        
                        if len(words) < 300:
                            st.warning("⚠️ Thin content (<300 words)")
                        elif len(words) > 2000:
                            st.success("✅ Comprehensive (>2000 words)")
                        else:
                            st.info(f"ℹ️ Medium length ({len(words)} words)")
                    
                    with col2:
                        st.markdown("**Top Keywords:**")
                        from collections import Counter
                        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'this', 'that', 'it', 'you', 'your', 'we', 'our', 'they', 'their', 'i', 'my', 'me', 'as', 'from', 'not', 'can', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
                        clean_words = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 3 and w.isalpha()]
                        top_words = Counter(clean_words).most_common(8)
                        for word, count in top_words:
                            pct = (count / len(words)) * 100
                            st.markdown(f"- **{word}**: {count} ({pct:.1f}%)")
                else:
                    st.warning("No body content extracted for this page.")
            
            with tab3:
                st.subheader("🔗 Outgoing Internal Links")
                cursor.execute('''
                    SELECT target_url, anchor_text, context, similarity_score 
                    FROM internal_links WHERE source_url = ?
                ''', (selected_url,))
                outgoing = cursor.fetchall()
                
                if outgoing:
                    df = pd.DataFrame(outgoing, columns=['Target URL', 'Anchor Text', 'Context', 'Similarity'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No outgoing internal links")
                
                st.divider()
                
                st.subheader("🔗 Incoming Internal Links")
                cursor.execute('''
                    SELECT source_url, anchor_text, context, similarity_score 
                    FROM internal_links WHERE target_url = ?
                ''', (selected_url,))
                incoming = cursor.fetchall()
                
                if incoming:
                    df = pd.DataFrame(incoming, columns=['Source URL', 'Anchor Text', 'Context', 'Similarity'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("⚠️ No incoming links - this is an orphan page!")
            
            with tab4:
                st.subheader("🤖 AI-Powered Analysis")
                
                active_provider = ai_provider.get_active_provider()
                
                if not active_provider:
                    st.warning("⚠️ No AI provider configured. Go to '🔌 AI Settings' to add an API key.")
                    st.info("Supported providers: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini)")
                else:
                    st.success(f"✅ Using: **{active_provider['provider'].upper()}** ({active_provider['model']})")
                    
                    if st.button("🔍 Analyze with AI", type="primary"):
                        with st.spinner("Analyzing content with AI..."):
                            prompt = f"""Analyze this webpage content for SEO and internal linking:

Title: {page['meta_title']}
Description: {page['meta_description']}
H1: {page['h1']}

Content:
{(page['body_content'] or '')[:3500]}

Provide a concise analysis:
1. Main topics (3-5 keywords)
2. Content quality score (1-10) with reasons
3. Internal linking opportunities (what topics to link to)
4. SEO recommendations (2-3 actionable items)
"""
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                analysis = loop.run_until_complete(
                                    ai_provider.call_ai(prompt, active_provider)
                                )
                            finally:
                                loop.close()
                            
                            st.markdown("### 📋 Analysis Results")
                            st.markdown(analysis)
    
    return None


def show_clusters_embeddings(conn, ai_provider):
    """Clusters & Embeddings page"""
    st.markdown('<p class="main-header">🧠 Clusters & Embeddings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Semantic analysis with grouped URLs and similarity scores</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pages WHERE crawl_status = 'success'")
    page_count = cursor.fetchone()[0]
    
    if page_count == 0:
        st.warning("No pages crawled yet. Go to 'Crawl URLs' first.")
        return None
    
    tab1, tab2, tab3 = st.tabs(["🔄 Generate Embeddings", "📊 Similarity Matrix", "🎯 Grouped Links"])
    
    with tab1:
        st.subheader("Generate Embeddings & Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_method = st.selectbox("Clustering Method", ['hdbscan', 'kmeans'])
            if cluster_method == 'hdbscan':
                min_cluster_size = st.slider("Min Cluster Size", 2, 20, 5)
            else:
                n_clusters = st.slider("Number of Clusters", 2, 30, 10)
        
        with col2:
            dim_method = st.selectbox("Visualization Method", ['umap', 'tsne'])
        
        if st.button("🚀 Generate Embeddings & Clusters", type="primary"):
            with st.spinner("Processing embeddings..."):
                cursor.execute("SELECT url, meta_title, h1, body_content FROM pages WHERE crawl_status = 'success'")
                pages = cursor.fetchall()
                
                if len(pages) < 3:
                    st.error("Need at least 3 pages to cluster")
                    return None
                
                # Prepare texts
                texts = []
                urls = []
                for p in pages:
                    text = f"{p[1] or ''} {p[2] or ''} {(p[3] or '')[:1500]}"
                    texts.append(text)
                    urls.append(p[0])
                
                progress = st.progress(0)
                status = st.empty()
                
                # Generate embeddings
                status.text("Generating embeddings...")
                progress.progress(20)
                embeddings = generate_embeddings(texts)
                
                # Calculate similarity
                status.text("Calculating similarities...")
                progress.progress(40)
                sim_matrix = calculate_similarity_matrix(embeddings)
                
                # Cluster
                status.text("Clustering pages...")
                progress.progress(60)
                if cluster_method == 'hdbscan':
                    labels = cluster_pages(embeddings, 'hdbscan', min_cluster_size=min_cluster_size)
                else:
                    labels = cluster_pages(embeddings, 'kmeans', n_clusters=n_clusters)
                
                # Reduce dimensions
                status.text("Reducing dimensions...")
                progress.progress(80)
                coords = reduce_dimensions(embeddings, dim_method)
                
                # Save to database
                status.text("Saving results...")
                progress.progress(90)
                
                for i, url in enumerate(urls):
                    cluster_name = f"Cluster {labels[i]}" if labels[i] >= 0 else "Unclustered"
                    cursor.execute('''
                        UPDATE pages SET embedding = ?, cluster_id = ?, cluster_name = ?
                        WHERE url = ?
                    ''', (embeddings[i].tobytes(), int(labels[i]), cluster_name, url))
                    
                    # Update link similarity scores
                    for j, other_url in enumerate(urls):
                        if i != j:
                            cursor.execute('''
                                UPDATE internal_links SET similarity_score = ?
                                WHERE source_url = ? AND target_url = ?
                            ''', (float(sim_matrix[i][j]), url, other_url))
                
                conn.commit()
                progress.progress(100)
                status.empty()
                
                st.success(f"✅ Generated embeddings for {len(urls)} pages!")
                
                # Visualization
                st.subheader("📊 Cluster Visualization")
                
                df_viz = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'url': urls,
                    'cluster': [f"Cluster {l}" if l >= 0 else "Noise" for l in labels],
                    'title': [p[1][:40] if p[1] else p[0].split('/')[-1][:40] for p in pages]
                })
                
                fig = px.scatter(
                    df_viz, x='x', y='y', color='cluster',
                    hover_data=['title', 'url'],
                    title="Page Clusters (2D Projection)"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Similarity Matrix")
        
        cursor.execute('''
            SELECT url, meta_title, embedding FROM pages 
            WHERE crawl_status = 'success' AND embedding IS NOT NULL
            LIMIT 50
        ''')
        pages_with_emb = cursor.fetchall()
        
        if not pages_with_emb:
            st.warning("Generate embeddings first in the 'Generate Embeddings' tab.")
        else:
            urls = [p[0] for p in pages_with_emb]
            titles = [p[1] or p[0].split('/')[-1] for p in pages_with_emb]
            embeddings = np.array([np.frombuffer(p[2], dtype=np.float32) for p in pages_with_emb])
            
            sim_matrix = calculate_similarity_matrix(embeddings)
            
            short_titles = [t[:25] + '...' if len(t) > 25 else t for t in titles]
            
            fig = go.Figure(data=go.Heatmap(
                z=sim_matrix,
                x=short_titles,
                y=short_titles,
                colorscale='RdYlGn',
                hoverongaps=False
            ))
            fig.update_layout(title="Page Similarity Matrix", height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top similar pairs
            st.subheader("🔝 Most Similar Page Pairs")
            pairs = []
            for i in range(len(urls)):
                for j in range(i + 1, len(urls)):
                    pairs.append({
                        'Page 1': titles[i][:35],
                        'Page 2': titles[j][:35],
                        'Similarity': round(sim_matrix[i][j], 3)
                    })
            
            pairs_df = pd.DataFrame(pairs).sort_values('Similarity', ascending=False).head(15)
            st.dataframe(pairs_df, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Grouped Links by Similarity")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_similarity = st.slider("Min Similarity Score", 0.0, 1.0, 0.3, 0.05)
        
        with col2:
            cursor.execute("SELECT DISTINCT source_url FROM internal_links")
            source_urls = ['All'] + [r[0] for r in cursor.fetchall()]
            filter_source = st.selectbox("Filter by Source URL", source_urls)
        
        with col3:
            cursor.execute("SELECT DISTINCT target_url FROM internal_links")
            target_urls = ['All'] + [r[0] for r in cursor.fetchall()]
            filter_target = st.selectbox("Filter by Target URL", target_urls)
        
        # Build query
        query = '''
            SELECT il.source_url, il.target_url, il.anchor_text, il.similarity_score,
                   p1.meta_title as source_title, p2.meta_title as target_title,
                   p1.cluster_name as source_cluster, p2.cluster_name as target_cluster
            FROM internal_links il
            LEFT JOIN pages p1 ON il.source_url = p1.url
            LEFT JOIN pages p2 ON il.target_url = p2.url
            WHERE il.similarity_score >= ?
        '''
        params = [min_similarity]
        
        if filter_source != 'All':
            query += ' AND il.source_url = ?'
            params.append(filter_source)
        
        if filter_target != 'All':
            query += ' AND il.target_url = ?'
            params.append(filter_target)
        
        query += ' ORDER BY il.similarity_score DESC LIMIT 200'
        
        cursor.execute(query, params)
        links = cursor.fetchall()
        
        if links:
            # Group by source URL
            grouped = {}
            for link in links:
                source = link[0]
                if source not in grouped:
                    grouped[source] = {
                        'source_title': link[4] or source.split('/')[-1],
                        'source_cluster': link[6],
                        'links': []
                    }
                grouped[source]['links'].append({
                    'target_url': link[1],
                    'target_title': link[5] or link[1].split('/')[-1],
                    'anchor_text': link[2],
                    'similarity': link[3] or 0,
                    'target_cluster': link[7]
                })
            
            # Display
            for source_url, data in grouped.items():
                with st.expander(f"📄 {data['source_title'][:50]}... ({len(data['links'])} links)"):
                    st.markdown(f"**Source URL:** [{source_url[:60]}...]({source_url})")
                    st.markdown(f"**Cluster:** {data['source_cluster'] or 'Not assigned'}")
                    st.divider()
                    
                    for link in data['links']:
                        sim = link['similarity']
                        sim_class = 'high' if sim > 0.7 else ('medium' if sim > 0.5 else 'low')
                        
                        st.markdown(f"""
                        <div class="link-group">
                            <strong>→ {link['target_title'][:45]}...</strong><br>
                            <small>URL: {link['target_url'][:50]}...</small><br>
                            <small>Anchor: "{(link['anchor_text'] or 'N/A')[:35]}..."</small><br>
                            <span class="similarity-{sim_class}">Similarity: {sim:.3f}</span>
                            | Cluster: {link['target_cluster'] or 'N/A'}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Export
            st.divider()
            export_data = [{
                'Source URL': l[0],
                'Source Title': l[4],
                'Target URL': l[1],
                'Target Title': l[5],
                'Anchor Text': l[2],
                'Similarity': l[3],
                'Source Cluster': l[6],
                'Target Cluster': l[7]
            } for l in links]
            
            csv = pd.DataFrame(export_data).to_csv(index=False)
            st.download_button("📥 Export Grouped Links (CSV)", csv, "grouped_links.csv", "text/csv")
        else:
            st.info("No links found matching filters. Try lowering the similarity threshold.")
    
    return None


def show_link_suggestions(conn, ai_provider):
    """Link Suggestions page"""
    st.markdown('<p class="main-header">💡 Link Suggestions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered internal linking opportunities</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    
    tab1, tab2 = st.tabs(["🔄 Generate Suggestions", "📋 View Suggestions"])
    
    with tab1:
        st.subheader("Generate New Link Suggestions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Suggestions per page", 1, 20, 5)
            min_threshold = st.slider("Min similarity threshold", 0.0, 1.0, 0.3, 0.05)
        
        with col2:
            active_provider = ai_provider.get_active_provider()
            use_ai = st.checkbox(
                "Use AI for detailed suggestions",
                value=False,
                disabled=not active_provider,
                help="AI will provide explanations and better anchor text"
            )
            
            if active_provider:
                st.success(f"AI Provider: {active_provider['provider'].upper()}")
            else:
                st.info("Configure AI in 'AI Settings' for AI-powered suggestions")
        
        if st.button("🚀 Generate Suggestions", type="primary"):
            with st.spinner("Generating suggestions..."):
                # Get pages with embeddings
                cursor.execute('''
                    SELECT url, meta_title, body_content, embedding FROM pages 
                    WHERE crawl_status = 'success' AND embedding IS NOT NULL
                ''')
                pages = cursor.fetchall()
                
                if len(pages) < 2:
                    st.error("Need at least 2 pages with embeddings. Generate embeddings first.")
                    return None
                
                # Get existing links
                cursor.execute('SELECT source_url, target_url FROM internal_links')
                existing_links = set((r[0], r[1]) for r in cursor.fetchall())
                
                # Prepare data
                urls = [p[0] for p in pages]
                titles = [p[1] for p in pages]
                contents = [p[2] for p in pages]
                embeddings = np.array([np.frombuffer(p[3], dtype=np.float32) for p in pages])
                
                # Calculate similarity
                sim_matrix = calculate_similarity_matrix(embeddings)
                
                # Generate suggestions
                suggestions = []
                progress = st.progress(0)
                
                for i, source_url in enumerate(urls):
                    progress.progress((i + 1) / len(urls))
                    
                    # Get top similar pages
                    similarities = sim_matrix[i]
                    top_indices = np.argsort(similarities)[::-1]
                    
                    count = 0
                    for j in top_indices:
                        if count >= top_k:
                            break
                        
                        if i == j:
                            continue
                        
                        target_url = urls[j]
                        
                        # Skip existing links
                        if (source_url, target_url) in existing_links:
                            continue
                        
                        similarity = similarities[j]
                        
                        if similarity < min_threshold:
                            break
                        
                        # Determine priority
                        if similarity >= 0.7:
                            priority = 'high'
                        elif similarity >= 0.5:
                            priority = 'medium'
                        else:
                            priority = 'low'
                        
                        # Generate anchor text
                        anchor = titles[j] if titles[j] else target_url.split('/')[-1]
                        if len(anchor) > 60:
                            anchor = anchor[:60]
                        
                        # AI explanation (if enabled)
                        ai_explanation = ''
                        if use_ai and active_provider and count < 2:
                            prompt = f"""Suggest the best way to link from one page to another:

Source page topic: {(contents[i] or '')[:500]}
Target page topic: {(contents[j] or '')[:500]}

Provide brief: 1) Best anchor text, 2) Why link these pages (1 sentence)"""
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                ai_explanation = loop.run_until_complete(
                                    ai_provider.call_ai(prompt, active_provider)
                                )
                            finally:
                                loop.close()
                        
                        suggestions.append({
                            'source_url': source_url,
                            'target_url': target_url,
                            'anchor': anchor,
                            'similarity': similarity,
                            'priority': priority,
                            'ai_explanation': ai_explanation
                        })
                        
                        count += 1
                
                # Save suggestions
                cursor.execute('DELETE FROM suggestions')
                
                for s in suggestions:
                    cursor.execute('''
                        INSERT OR REPLACE INTO suggestions 
                        (source_url, target_url, suggested_anchor, relevance_score, priority, ai_explanation, status)
                        VALUES (?, ?, ?, ?, ?, ?, 'pending')
                    ''', (s['source_url'], s['target_url'], s['anchor'], s['similarity'], s['priority'], s['ai_explanation']))
                
                conn.commit()
                st.success(f"✅ Generated {len(suggestions)} link suggestions!")
    
    with tab2:
        st.subheader("Link Suggestions")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority_filter = st.selectbox("Priority", ['All', 'high', 'medium', 'low'])
        
        with col2:
            status_filter = st.selectbox("Status", ['All', 'pending', 'approved', 'rejected'])
        
        with col3:
            sort_by = st.selectbox("Sort by", ['Score (High to Low)', 'Score (Low to High)'])
        
        # Query
        query = 'SELECT * FROM suggestions WHERE 1=1'
        params = []
        
        if priority_filter != 'All':
            query += ' AND priority = ?'
            params.append(priority_filter)
        
        if status_filter != 'All':
            query += ' AND status = ?'
            params.append(status_filter)
        
        query += ' ORDER BY relevance_score ' + ('DESC' if 'High to Low' in sort_by else 'ASC')
        query += ' LIMIT 100'
        
        cursor.execute(query, params)
        suggestions = cursor.fetchall()
        
        if suggestions:
            for s in suggestions:
                s_dict = dict(s)
                
                # Safe priority icon lookup - handle None values
                priority_value = s_dict.get('priority') or ''
                priority_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority_value, '⚪')
                
                # Safe URL display
                source_url = s_dict.get('source_url') or ''
                target_url = s_dict.get('target_url') or ''
                
                with st.expander(f"{priority_icon} {source_url[:40]}... → {target_url[:40]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Source:** [{source_url}]({source_url})")
                        st.markdown(f"**Target:** [{target_url}]({target_url})")
                        st.markdown(f"**Suggested Anchor:** `{s_dict.get('suggested_anchor') or 'N/A'}`")
                        st.markdown(f"**Relevance Score:** {safe_score_format(s_dict.get('relevance_score'))}")
                        
                        if s_dict.get('ai_explanation'):
                            st.markdown("**AI Explanation:**")
                            st.info(s_dict['ai_explanation'])
                    
                    with col2:
                        priority_display = (s_dict.get('priority') or 'N/A').upper()
                        st.markdown(f"**Priority:** {priority_display}")
                        st.markdown(f"**Status:** {s_dict.get('status') or 'N/A'}")
                        
                        if s_dict.get('status') == 'pending':
                            suggestion_id = s_dict.get('id')
                            if suggestion_id:
                                if st.button("✅ Approve", key=f"approve_{suggestion_id}"):
                                    cursor.execute('UPDATE suggestions SET status = ? WHERE id = ?', ('approved', suggestion_id))
                                    conn.commit()
                                    st.rerun()
                                
                                if st.button("❌ Reject", key=f"reject_{suggestion_id}"):
                                    cursor.execute('UPDATE suggestions SET status = ? WHERE id = ?', ('rejected', suggestion_id))
                                    conn.commit()
                                    st.rerun()
            
            # Export
            st.divider()
            df_export = pd.DataFrame([dict(s) for s in suggestions])
            csv = df_export.to_csv(index=False)
            st.download_button("📥 Export Suggestions (CSV)", csv, "link_suggestions.csv", "text/csv")
        else:
            st.info("No suggestions found. Generate suggestions first or adjust filters.")
    
    return None


def show_ai_settings(conn, ai_provider):
    """AI Settings page - FIXED"""
    st.markdown('<p class="main-header">🔌 AI Settings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure AI providers for enhanced analysis</p>', unsafe_allow_html=True)
    
    # Current status
    active_provider = ai_provider.get_active_provider()
    
    if active_provider:
        st.success(f"✅ **Active Provider:** {active_provider['provider'].upper()} ({active_provider['model']})")
    else:
        st.warning("⚠️ No AI provider configured yet. Add an API key below to enable AI features.")
    
    st.divider()
    
    # Add new API key
    st.subheader("➕ Add API Key")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 OpenAI (GPT-4)")
        st.markdown("*Best for general analysis*")
        
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key_input", placeholder="sk-...")
        openai_model = st.selectbox(
            "Model",
            ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-3.5-turbo'],
            key="openai_model_select"
        )
        
        if st.button("💾 Save OpenAI", key="save_openai_btn", type="primary"):
            if openai_key and openai_key.startswith('sk-'):
                ai_provider.save_api_key('openai', openai_key, openai_model, True)
                st.success("✅ OpenAI API key saved and activated!")
                st.rerun()
            else:
                st.error("Please enter a valid OpenAI API key (starts with 'sk-')")
    
    with col2:
        st.markdown("### 🧠 Anthropic (Claude)")
        st.markdown("*Best for detailed analysis*")
        
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key_input", placeholder="sk-ant-...")
        anthropic_model = st.selectbox(
            "Model",
            ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20240620'],
            key="anthropic_model_select"
        )
        
        if st.button("💾 Save Anthropic", key="save_anthropic_btn", type="primary"):
            if anthropic_key and 'sk-ant' in anthropic_key:
                ai_provider.save_api_key('anthropic', anthropic_key, anthropic_model, True)
                st.success("✅ Anthropic API key saved and activated!")
                st.rerun()
            else:
                st.error("Please enter a valid Anthropic API key")
    
    with col3:
        st.markdown("### 💎 Google (Gemini)")
        st.markdown("*Free tier available*")
        
        google_key = st.text_input("Google API Key", type="password", key="google_key_input", placeholder="AIza...")
        google_model = st.selectbox(
            "Model",
            ['gemini-pro', 'gemini-1.5-pro', 'gemini-1.5-flash'],
            key="google_model_select"
        )
        
        if st.button("💾 Save Google", key="save_google_btn", type="primary"):
            if google_key and google_key.startswith('AIza'):
                ai_provider.save_api_key('google', google_key, google_model, True)
                st.success("✅ Google API key saved and activated!")
                st.rerun()
            else:
                st.error("Please enter a valid Google API key (starts with 'AIza')")
    
    st.divider()
    
    # Saved providers
    st.subheader("📋 Saved API Keys")
    
    all_providers = ai_provider.get_all_providers()
    
    if all_providers:
        for provider in all_providers:
            p = dict(provider)
            is_active = p['is_active'] == 1
            
            card_class = "api-active" if is_active else "api-inactive"
            status_icon = "🟢" if is_active else "⚪"
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{status_icon} {p['provider'].upper()}**")
            
            with col2:
                st.markdown(f"Model: `{p['model']}`")
            
            with col3:
                key_preview = p['api_key'][:8] + '...' + p['api_key'][-4:] if p['api_key'] else 'Not set'
                st.markdown(f"Key: `{key_preview}`")
            
            with col4:
                if not is_active:
                    if st.button(f"Activate", key=f"activate_{p['provider']}"):
                        ai_provider.set_active(p['provider'])
                        st.rerun()
                else:
                    st.markdown("**Active**")
            
            st.divider()
    else:
        st.info("No API keys saved yet. Add one above to get started.")
    
    # Help section
    st.subheader("❓ How to Get API Keys")
    
    with st.expander("🤖 OpenAI API Key"):
        st.markdown("""
        1. Go to [platform.openai.com](https://platform.openai.com)
        2. Sign up or log in
        3. Click on your profile → **API Keys**
        4. Click **Create new secret key**
        5. Copy and paste it above
        
        **Pricing:** Pay-per-use, ~$0.002/1K tokens for GPT-3.5, ~$0.03/1K for GPT-4
        """)
    
    with st.expander("🧠 Anthropic API Key"):
        st.markdown("""
        1. Go to [console.anthropic.com](https://console.anthropic.com)
        2. Sign up or log in
        3. Navigate to **API Keys**
        4. Click **Create Key**
        5. Copy and paste it above
        
        **Pricing:** Pay-per-use, starting at ~$0.008/1K tokens
        """)
    
    with st.expander("💎 Google Gemini API Key"):
        st.markdown("""
        1. Go to [makersuite.google.com](https://makersuite.google.com) or [aistudio.google.com](https://aistudio.google.com)
        2. Sign in with Google
        3. Click **Get API Key**
        4. Create a new key
        5. Copy and paste it above
        
        **Pricing:** Free tier available (60 requests/minute), then pay-per-use
        """)
    
    return None


def show_import_export(conn):
    """Import/Export page"""
    st.markdown('<p class="main-header">📥 Import/Export</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Manage your data</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Export Data")
        
        export_type = st.selectbox("What to export", ['All Pages', 'Internal Links', 'Suggestions', 'Everything (JSON)'])
        
        if st.button("📥 Generate Export", type="primary"):
            if export_type == 'All Pages':
                cursor.execute('SELECT url, meta_title, meta_description, h1, word_count, internal_links_count, cluster_name FROM pages WHERE crawl_status = "success"')
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=['URL', 'Title', 'Description', 'H1', 'Words', 'Internal Links', 'Cluster'])
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "pages_export.csv", "text/csv")
            
            elif export_type == 'Internal Links':
                cursor.execute('SELECT source_url, target_url, anchor_text, similarity_score FROM internal_links')
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=['Source', 'Target', 'Anchor Text', 'Similarity'])
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "links_export.csv", "text/csv")
            
            elif export_type == 'Suggestions':
                cursor.execute('SELECT source_url, target_url, suggested_anchor, relevance_score, priority, status FROM suggestions')
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=['Source', 'Target', 'Anchor', 'Score', 'Priority', 'Status'])
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "suggestions_export.csv", "text/csv")
            
            else:
                export_data = {}
                
                cursor.execute('SELECT url, meta_title, word_count, cluster_name FROM pages WHERE crawl_status = "success"')
                export_data['pages'] = [{'url': r[0], 'title': r[1], 'words': r[2], 'cluster': r[3]} for r in cursor.fetchall()]
                
                cursor.execute('SELECT source_url, target_url, anchor_text, similarity_score FROM internal_links')
                export_data['links'] = [{'source': r[0], 'target': r[1], 'anchor': r[2], 'similarity': r[3]} for r in cursor.fetchall()]
                
                cursor.execute('SELECT source_url, target_url, suggested_anchor, relevance_score, priority, status FROM suggestions')
                export_data['suggestions'] = [{'source': r[0], 'target': r[1], 'anchor': r[2], 'score': r[3], 'priority': r[4], 'status': r[5]} for r in cursor.fetchall()]
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button("Download JSON", json_str, "full_export.json", "application/json")
    
    with col2:
        st.subheader("📥 Import Data")
        
        uploaded_file = st.file_uploader("Upload file (CSV or JSON)", type=['csv', 'json'])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(), use_container_width=True)
                st.info("CSV preview above. Use 'Crawl URLs' page to import URLs for crawling.")
            else:
                data = json.load(uploaded_file)
                st.json(data)
        
        st.divider()
        
        st.subheader("🗑️ Clear Data")
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            confirm = st.checkbox("I understand this will permanently delete all data")
            if confirm:
                cursor.execute('DELETE FROM pages')
                cursor.execute('DELETE FROM internal_links')
                cursor.execute('DELETE FROM suggestions')
                conn.commit()
                st.success("✅ All data cleared!")
                st.rerun()
    
    return None


# ============ MAIN FUNCTION ============

def main():
    """Main application entry point"""
    # Initialize database
    conn = init_database()
    ai_provider = AIProvider(conn)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🔗 Internal Link Builder")
        st.markdown("*AI-Powered SEO Tool v2.0*")
        st.divider()
        
        # Navigation menu
        page = st.radio(
            "Navigation",
            [
                "📊 Dashboard",
                "🔍 Crawl URLs",
                "📄 Page Analysis",
                "🧠 Clusters & Embeddings",
                "💡 Link Suggestions",
                "🔌 AI Settings",
                "📥 Import/Export"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick stats
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM pages WHERE crawl_status = 'success'")
        pages_count = cursor.fetchone()[0]
        st.metric("📄 Pages", pages_count)
        
        cursor.execute("SELECT COUNT(*) FROM internal_links")
        links_count = cursor.fetchone()[0]
        st.metric("🔗 Links", links_count)
        
        cursor.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'")
        suggestions_count = cursor.fetchone()[0]
        st.metric("💡 Suggestions", suggestions_count)
        
        # AI Status
        st.divider()
        active = ai_provider.get_active_provider()
        if active:
            st.success(f"🤖 AI: {active['provider'].upper()}")
        else:
            st.warning("🤖 AI: Not configured")
            st.caption("Go to AI Settings →")
    
    # Route to selected page
    if page == "📊 Dashboard":
        show_dashboard(conn)
    elif page == "🔍 Crawl URLs":
        show_crawl_page(conn)
    elif page == "📄 Page Analysis":
        show_page_analysis(conn, ai_provider)
    elif page == "🧠 Clusters & Embeddings":
        show_clusters_embeddings(conn, ai_provider)
    elif page == "💡 Link Suggestions":
        show_link_suggestions(conn, ai_provider)
    elif page == "🔌 AI Settings":
        show_ai_settings(conn, ai_provider)
    elif page == "📥 Import/Export":
        show_import_export(conn)
    
    # Don't return anything from main()
    return None


# Run the app
if __name__ == "__main__":
    main()
