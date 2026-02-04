"""
Internal Link Builder - AI-Powered SEO Tool
Version 2.0 with AI API Integration (OpenAI, Gemini, Claude)
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
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import os

# Page config
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
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; }
    .content-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; max-height: 400px; overflow-y: auto; }
    .link-group { background-color: #e7f3ff; border-left: 4px solid #0066cc; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }
    .similarity-high { color: #28a745; font-weight: bold; }
    .similarity-medium { color: #ffc107; font-weight: bold; }
    .similarity-low { color: #dc3545; font-weight: bold; }
    .body-content-preview { background-color: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; font-size: 0.9rem; line-height: 1.6; max-height: 500px; overflow-y: auto; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)


# ============ DATABASE FUNCTIONS ============

def init_database():
    """Initialize SQLite database with all tables"""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/internal_links.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL, domain TEXT,
        meta_title TEXT, meta_description TEXT, h1 TEXT, h2_tags TEXT, body_content TEXT,
        word_count INTEGER DEFAULT 0, internal_links_count INTEGER DEFAULT 0,
        external_links_count INTEGER DEFAULT 0, crawl_status TEXT DEFAULT 'pending',
        crawl_error TEXT, http_status INTEGER, crawled_at TIMESTAMP,
        embedding BLOB, cluster_id INTEGER, cluster_name TEXT, page_rank REAL DEFAULT 0.0)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS internal_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT, source_url TEXT NOT NULL, target_url TEXT NOT NULL,
        anchor_text TEXT, context TEXT, is_dofollow INTEGER DEFAULT 1,
        similarity_score REAL DEFAULT 0.0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(source_url, target_url, anchor_text))''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, source_url TEXT NOT NULL, target_url TEXT NOT NULL,
        suggested_anchor TEXT, suggested_context TEXT, relevance_score REAL,
        priority TEXT DEFAULT 'medium', reason TEXT, ai_explanation TEXT,
        status TEXT DEFAULT 'pending', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(source_url, target_url))''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS api_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT UNIQUE, api_key TEXT,
        model TEXT, is_active INTEGER DEFAULT 0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    return conn


# ============ AI API FUNCTIONS ============

class AIProvider:
    def __init__(self, conn):
        self.conn = conn
    
    def get_active_provider(self) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM api_settings WHERE is_active = 1 LIMIT 1')
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def save_api_key(self, provider: str, api_key: str, model: str, is_active: bool = True):
        cursor = self.conn.cursor()
        if is_active:
            cursor.execute('UPDATE api_settings SET is_active = 0')
        cursor.execute('''INSERT OR REPLACE INTO api_settings (provider, api_key, model, is_active, updated_at)
            VALUES (?, ?, ?, ?, ?)''', (provider, api_key, model, 1 if is_active else 0, datetime.now()))
        self.conn.commit()
    
    def get_all_api_settings(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM api_settings')
        return [dict(row) for row in cursor.fetchall()]
    
    async def analyze_content_with_ai(self, content: str, provider_settings: Dict) -> str:
        provider = provider_settings['provider']
        api_key = provider_settings['api_key']
        model = provider_settings['model']
        
        prompt = f"""Analyze this webpage content for SEO and internal linking:

{content[:4000]}

Provide:
1. Main topics (3-5 keywords)
2. Content quality score (1-10)
3. Internal linking opportunities
4. SEO recommendations

Be concise and actionable."""

        try:
            if provider == 'openai':
                return await self._call_openai(prompt, api_key, model)
            elif provider == 'anthropic':
                return await self._call_anthropic(prompt, api_key, model)
            elif provider == 'google':
                return await self._call_google(prompt, api_key, model)
        except Exception as e:
            return f"Error: {str(e)}"
        return "AI analysis not available"
    
    async def generate_link_suggestion_ai(self, source: str, target: str, provider_settings: Dict) -> Dict:
        prompt = f"""Analyze these pages for internal linking:

SOURCE PAGE: {source[:1500]}
TARGET PAGE: {target[:1500]}

Return JSON with: relevance_score (0-100), anchor_text, explanation, seo_benefit"""

        try:
            provider = provider_settings['provider']
            if provider == 'openai':
                response = await self._call_openai(prompt, provider_settings['api_key'], provider_settings['model'])
            elif provider == 'anthropic':
                response = await self._call_anthropic(prompt, provider_settings['api_key'], provider_settings['model'])
            elif provider == 'google':
                response = await self._call_google(prompt, provider_settings['api_key'], provider_settings['model'])
            else:
                return {}
            
            try:
                return json.loads(response)
            except:
                return {'explanation': response}
        except Exception as e:
            return {'error': str(e)}
    
    async def _call_openai(self, prompt: str, api_key: str, model: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], max_tokens=1500)
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    async def _call_anthropic(self, prompt: str, api_key: str, model: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(model=model, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text
        except Exception as e:
            return f"Anthropic Error: {str(e)}"
    
    async def _call_google(self, prompt: str, api_key: str, model: str) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Google Error: {str(e)}"


# ============ CRAWLER FUNCTIONS ============

USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
}


def extract_body_content(soup: BeautifulSoup) -> str:
    """Extract clean body content from HTML"""
    for tag in soup.find_all(['script', 'style', 'noscript', 'iframe', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    
    main_content = None
    for selector in ['main', 'article', '[role="main"]', '.content', '.post-content', '#content']:
        main_content = soup.select_one(selector) if selector.startswith(('.', '#', '[')) else soup.find(selector)
        if main_content:
            break
    
    if not main_content:
        main_content = soup.find('body')
    
    if main_content:
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td', 'th'])
        text_parts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 15]
        return '\n\n'.join(text_parts)
    return ''


async def crawl_url(session: aiohttp.ClientSession, url: str, user_agent: str) -> Dict:
    result = {'url': url, 'status': 'pending', 'meta_title': '', 'meta_description': '', 'h1': '',
              'h2_tags': [], 'body_content': '', 'word_count': 0, 'internal_links': [], 'external_links': [], 'error': ''}
    
    try:
        headers = {'User-Agent': USER_AGENTS.get(user_agent, USER_AGENTS['googlebot'])}
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), ssl=False) as response:
            result['http_status'] = response.status
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                domain = urlparse(url).netloc
                
                title_tag = soup.find('title')
                result['meta_title'] = title_tag.get_text(strip=True) if title_tag else ''
                
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                result['meta_description'] = meta_desc.get('content', '') if meta_desc else ''
                
                h1_tag = soup.find('h1')
                result['h1'] = h1_tag.get_text(strip=True) if h1_tag else ''
                
                result['h2_tags'] = [h2.get_text(strip=True) for h2 in soup.find_all('h2')[:10]]
                
                result['body_content'] = extract_body_content(soup)
                result['word_count'] = len(result['body_content'].split())
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                        continue
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)
                    if not parsed.scheme or not parsed.netloc:
                        continue
                    
                    anchor_text = a_tag.get_text(strip=True)
                    parent = a_tag.find_parent(['p', 'li', 'div'])
                    context = parent.get_text(strip=True)[:200] if parent else ''
                    rel = a_tag.get('rel', [])
                    is_dofollow = 'nofollow' not in (rel if isinstance(rel, list) else [rel])
                    
                    link_data = {'target_url': full_url, 'anchor_text': anchor_text, 'context': context, 'is_dofollow': is_dofollow}
                    
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
        result['error'] = str(e)
    
    return result


async def crawl_urls(urls: List[str], user_agent: str, max_concurrent: int, delay: float, progress_callback=None) -> List[Dict]:
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_semaphore(url: str):
            async with semaphore:
                if delay > 0:
                    await asyncio.sleep(delay)
                return await crawl_url(session, url, user_agent)
        
        tasks = [crawl_with_semaphore(url) for url in urls]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(urls))
        return results


# ============ EMBEDDING & CLUSTERING ============

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(texts: List[str]) -> np.ndarray:
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=True)


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


def cluster_embeddings(embeddings: np.ndarray, method: str = 'hdbscan', **kwargs):
    if method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=kwargs.get('min_cluster_size', 5))
        return clusterer.fit_predict(embeddings)
    else:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=kwargs.get('n_clusters', 10), random_state=42).fit_predict(embeddings)


def reduce_dimensions(embeddings: np.ndarray, method: str = 'umap') -> np.ndarray:
    if method == 'umap':
        import umap
        return umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)).fit_transform(embeddings)


# ============ MAIN APPLICATION ============

def main():
    conn = init_database()
    ai_provider = AIProvider(conn)
    
    with st.sidebar:
        st.markdown("## 🔗 Internal Link Builder")
        st.markdown("*AI-Powered SEO Tool v2.0*")
        st.divider()
        
        page = st.radio("Navigation", ["📊 Dashboard", "🔍 Crawl URLs", "📄 Page Analysis", 
                        "🧠 Clusters & Embeddings", "💡 Link Suggestions", "🔌 AI Settings", "📥 Import/Export"],
                        label_visibility="collapsed")
        
        st.divider()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pages WHERE crawl_status = 'success'")
        st.metric("📄 Pages", cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM internal_links")
        st.metric("🔗 Links", cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'")
        st.metric("💡 Suggestions", cursor.fetchone()[0])
        
        active = ai_provider.get_active_provider()
        st.success(f"🤖 {active['provider'].upper()}") if active else st.warning("🤖 No AI")
    
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


def show_dashboard(conn):
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    cursor = conn.cursor()
    
    col1, col2, col3, col4 = st.columns(4)
    cursor.execute("SELECT COUNT(*) FROM pages WHERE crawl_status = 'success'")
    col1.metric("Pages Crawled", cursor.fetchone()[0])
    cursor.execute("SELECT COUNT(*) FROM internal_links")
    col2.metric("Internal Links", cursor.fetchone()[0])
    cursor.execute("SELECT COUNT(DISTINCT cluster_id) FROM pages WHERE cluster_id IS NOT NULL")
    col3.metric("Clusters", cursor.fetchone()[0])
    cursor.execute("SELECT COUNT(*) FROM suggestions WHERE status = 'pending'")
    col4.metric("Opportunities", cursor.fetchone()[0])
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Pages by Cluster")
        cursor.execute("SELECT COALESCE(cluster_name, 'Unclustered') as cluster, COUNT(*) as count FROM pages WHERE crawl_status = 'success' GROUP BY cluster_id ORDER BY count DESC LIMIT 10")
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['Cluster', 'Count'])
            fig = px.pie(df, values='Count', names='Cluster', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster data yet")
    
    with col2:
        st.subheader("🔗 Top Linked Pages")
        cursor.execute("SELECT target_url, COUNT(*) as links FROM internal_links GROUP BY target_url ORDER BY links DESC LIMIT 10")
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['URL', 'Links'])
            df['URL'] = df['URL'].apply(lambda x: x.split('/')[-1][:25] or 'home')
            fig = px.bar(df, x='Links', y='URL', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No link data yet")
    
    st.subheader("🚨 Orphan Pages")
    cursor.execute("SELECT url, meta_title, word_count FROM pages WHERE crawl_status = 'success' AND url NOT IN (SELECT DISTINCT target_url FROM internal_links) LIMIT 10")
    orphans = cursor.fetchall()
    if orphans:
        st.dataframe(pd.DataFrame(orphans, columns=['URL', 'Title', 'Words']), use_container_width=True)
    else:
        st.success("No orphan pages!")


def show_crawl_page(conn):
    st.markdown('<p class="main-header">🔍 Crawl URLs</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        urls_input = st.text_area("Enter URLs (one per line)", height=200, placeholder="https://example.com/page1\nhttps://example.com/page2")
    with col2:
        user_agent = st.selectbox("User Agent", ['googlebot', 'bingbot', 'chrome'])
        max_concurrent = st.slider("Concurrent", 1, 50, 10)
        delay = st.slider("Delay (s)", 0.0, 5.0, 0.5, 0.1)
    
    uploaded = st.file_uploader("Or upload CSV/TXT", type=['csv', 'txt'])
    if uploaded:
        content = uploaded.getvalue().decode('utf-8')
        urls_input = '\n'.join(pd.read_csv(uploaded).iloc[:, 0].tolist()) if uploaded.name.endswith('.csv') else content
    
    if st.button("🚀 Start Crawling", type="primary"):
        urls = [u.strip() for u in urls_input.strip().split('\n') if u.strip()]
        if not urls:
            st.error("Enter at least one URL")
            return
        
        progress = st.progress(0)
        status = st.empty()
        
        def update(current, total):
            progress.progress(current / total)
            status.text(f"Crawling: {current}/{total}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(crawl_urls(urls, user_agent, max_concurrent, delay, update))
        finally:
            loop.close()
        
        cursor = conn.cursor()
        success, errors = 0, 0
        
        for r in results:
            try:
                cursor.execute('''INSERT OR REPLACE INTO pages (url, domain, meta_title, meta_description, h1, h2_tags, body_content, word_count, internal_links_count, external_links_count, crawl_status, crawl_error, http_status, crawled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (r['url'], urlparse(r['url']).netloc, r['meta_title'], r['meta_description'], r['h1'],
                     json.dumps(r['h2_tags']), r['body_content'], r['word_count'], len(r['internal_links']),
                     len(r['external_links']), r['status'], r.get('error', ''), r.get('http_status', 0), datetime.now()))
                
                for link in r['internal_links']:
                    try:
                        cursor.execute('INSERT OR IGNORE INTO internal_links (source_url, target_url, anchor_text, context, is_dofollow) VALUES (?, ?, ?, ?, ?)',
                            (r['url'], link['target_url'], link['anchor_text'], link['context'], 1 if link['is_dofollow'] else 0))
                    except:
                        pass
                
                success += 1 if r['status'] == 'success' else 0
                errors += 1 if r['status'] == 'error' else 0
            except:
                errors += 1
        
        conn.commit()
        st.success(f"✅ Done! {success} succeeded, {errors} failed")
        
        df = pd.DataFrame([{'URL': r['url'], 'Status': r['status'], 'Title': r['meta_title'][:40], 'Words': r['word_count'], 'Links': len(r['internal_links'])} for r in results])
        st.dataframe(df, use_container_width=True)


def show_page_analysis(conn, ai_provider):
    st.markdown('<p class="main-header">📄 Page Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed analysis with full content extraction</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    cursor.execute("SELECT url, meta_title, word_count FROM pages WHERE crawl_status = 'success' ORDER BY crawled_at DESC")
    pages = cursor.fetchall()
    
    if not pages:
        st.warning("No pages yet. Crawl URLs first.")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_url = st.selectbox("Select page", [p[0] for p in pages], format_func=lambda x: x[:80])
    with col2:
        search = st.text_input("🔍 Filter", placeholder="Search...")
        if search:
            filtered = [p[0] for p in pages if search.lower() in p[0].lower()]
            if filtered:
                selected_url = st.selectbox("Filtered", filtered, key="f")
    
    if selected_url:
        cursor.execute('SELECT * FROM pages WHERE url = ?', (selected_url,))
        page = dict(cursor.fetchone())
        
        st.markdown(f"### 📄 {page['meta_title'] or 'Untitled'}")
        st.markdown(f"**URL:** [{selected_url}]({selected_url})")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📝 Content", "🔗 Links", "🤖 AI Analysis"])
        
        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words", page['word_count'])
            c2.metric("Internal Links", page['internal_links_count'])
            c3.metric("External Links", page['external_links_count'])
            c4.metric("Cluster", page['cluster_name'] or 'N/A')
            
            st.divider()
            st.subheader("🏷️ Meta Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Meta Title:**")
                st.info(page['meta_title'] or 'Not found')
                tlen = len(page['meta_title'] or '')
                st.warning(f"⚠️ Title: {tlen} chars") if tlen < 30 or tlen > 60 else st.success(f"✅ Good: {tlen} chars")
            
            with col2:
                st.markdown("**Meta Description:**")
                st.info(page['meta_description'] or 'Not found')
                dlen = len(page['meta_description'] or '')
                st.warning(f"⚠️ Desc: {dlen} chars") if dlen < 120 or dlen > 160 else st.success(f"✅ Good: {dlen} chars")
            
            st.markdown("**H1:**")
            st.info(page['h1'] or 'Not found')
            
            if page['h2_tags']:
                st.markdown("**H2 Tags:**")
                h2s = json.loads(page['h2_tags']) if isinstance(page['h2_tags'], str) else page['h2_tags']
                for h2 in h2s:
                    st.markdown(f"- {h2}")
        
        with tab2:
            st.subheader("📝 Page Content")
            body = page['body_content']
            
            if body:
                words = body.split()
                sentences = [s for s in re.split(r'[.!?]+', body) if s.strip()]
                paragraphs = [p for p in body.split('\n\n') if p.strip()]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Words", len(words))
                c2.metric("Sentences", len(sentences))
                c3.metric("Paragraphs", len(paragraphs))
                c4.metric("Avg Words/Sent", round(len(words) / max(len(sentences), 1), 1))
                
                st.divider()
                display = st.radio("Display", ["Full", "First 500 words", "First 200 words"], horizontal=True)
                
                if display == "Full":
                    content = body
                elif display == "First 500 words":
                    content = ' '.join(words[:500])
                else:
                    content = ' '.join(words[:200])
                
                st.markdown(f'<div class="body-content-preview">{content}</div>', unsafe_allow_html=True)
                st.download_button("📥 Download Content", body, f"content_{urlparse(selected_url).path.replace('/', '_')}.txt")
                
                st.divider()
                st.subheader("📊 Content Quality")
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
                    st.markdown(f"- Avg word length: {avg_word_len:.1f} chars")
                    st.markdown(f"- Long sentences: {sum(1 for s in sentences if len(s.split()) > 25)}")
                    
                    if len(words) < 300:
                        st.warning("⚠️ Thin content (<300 words)")
                    elif len(words) > 2000:
                        st.success("✅ Comprehensive (>2000 words)")
                    else:
                        st.info(f"ℹ️ Medium ({len(words)} words)")
                
                with col2:
                    from collections import Counter
                    stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'this', 'that', 'it', 'you', 'we', 'they', 'i', 'my', 'your', 'our', 'their'}
                    clean = [w.lower() for w in words if w.lower() not in stops and len(w) > 3]
                    for word, count in Counter(clean).most_common(8):
                        st.markdown(f"- **{word}**: {count} ({count/len(words)*100:.1f}%)")
            else:
                st.warning("No content extracted")
        
        with tab3:
            st.subheader("🔗 Outgoing Links")
            cursor.execute('SELECT target_url, anchor_text, context, similarity_score FROM internal_links WHERE source_url = ?', (selected_url,))
            outgoing = cursor.fetchall()
            if outgoing:
                df = pd.DataFrame(outgoing, columns=['Target', 'Anchor', 'Context', 'Similarity'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No outgoing links")
            
            st.subheader("🔗 Incoming Links")
            cursor.execute('SELECT source_url, anchor_text, context, similarity_score FROM internal_links WHERE target_url = ?', (selected_url,))
            incoming = cursor.fetchall()
            if incoming:
                df = pd.DataFrame(incoming, columns=['Source', 'Anchor', 'Context', 'Similarity'])
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("⚠️ No incoming links - orphan page!")
        
        with tab4:
            st.subheader("🤖 AI Analysis")
            active = ai_provider.get_active_provider()
            
            if not active:
                st.warning("Configure AI in Settings first")
            else:
                st.info(f"Using: {active['provider'].upper()} ({active['model']})")
                
                if st.button("🔍 Analyze with AI", type="primary"):
                    with st.spinner("Analyzing..."):
                        content = f"Title: {page['meta_title']}\nDesc: {page['meta_description']}\nH1: {page['h1']}\nContent: {page['body_content'][:3000]}"
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            analysis = loop.run_until_complete(ai_provider.analyze_content_with_ai(content, active))
                        finally:
                            loop.close()
                        
                        st.markdown("### Results:")
                        st.markdown(analysis)


def show_clusters_embeddings(conn, ai_provider):
    st.markdown('<p class="main-header">🧠 Clusters & Embeddings</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pages WHERE crawl_status = 'success'")
    if cursor.fetchone()[0] == 0:
        st.warning("Crawl URLs first")
        return
    
    tab1, tab2, tab3 = st.tabs(["🔄 Generate", "📊 Similarity", "🎯 Grouped Links"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cluster_method = st.selectbox("Clustering", ['hdbscan', 'kmeans'])
            if cluster_method == 'hdbscan':
                min_size = st.slider("Min Cluster Size", 2, 20, 5)
            else:
                n_clusters = st.slider("Clusters", 2, 30, 10)
        with col2:
            dim_method = st.selectbox("Visualization", ['umap', 'tsne'])
        
        if st.button("🚀 Generate Embeddings & Clusters", type="primary"):
            with st.spinner("Processing..."):
                cursor.execute("SELECT url, meta_title, h1, body_content FROM pages WHERE crawl_status = 'success'")
                pages = cursor.fetchall()
                
                texts = [f"{p[1] or ''} {p[2] or ''} {(p[3] or '')[:1500]}" for p in pages]
                urls = [p[0] for p in pages]
                
                prog = st.progress(0)
                prog.progress(20)
                embeddings = generate_embeddings(texts)
                
                prog.progress(40)
                sim_matrix = calculate_similarity_matrix(embeddings)
                
                prog.progress(60)
                labels = cluster_embeddings(embeddings, cluster_method, min_cluster_size=min_size if cluster_method == 'hdbscan' else 5, n_clusters=n_clusters if cluster_method == 'kmeans' else 10)
                
                prog.progress(80)
                coords = reduce_dimensions(embeddings, dim_method)
                
                prog.progress(90)
                for i, url in enumerate(urls):
                    name = f"Cluster {labels[i]}" if labels[i] >= 0 else "Unclustered"
                    cursor.execute('UPDATE pages SET embedding = ?, cluster_id = ?, cluster_name = ? WHERE url = ?',
                        (embeddings[i].tobytes(), int(labels[i]), name, url))
                    for j, other in enumerate(urls):
                        if i != j:
                            cursor.execute('UPDATE internal_links SET similarity_score = ? WHERE source_url = ? AND target_url = ?',
                                (float(sim_matrix[i][j]), url, other))
                
                conn.commit()
                prog.progress(100)
                st.success(f"✅ Generated for {len(urls)} pages!")
                
                df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'url': urls,
                    'cluster': [f"Cluster {l}" if l >= 0 else "Noise" for l in labels],
                    'title': [p[1][:40] if p[1] else p[0].split('/')[-1] for p in pages]})
                
                fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['title', 'url'], title="Clusters")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        cursor.execute("SELECT url, meta_title, embedding FROM pages WHERE crawl_status = 'success' AND embedding IS NOT NULL LIMIT 50")
        pages = cursor.fetchall()
        
        if not pages:
            st.warning("Generate embeddings first")
        else:
            urls = [p[0] for p in pages]
            titles = [p[1] or p[0].split('/')[-1] for p in pages]
            embeddings = np.array([np.frombuffer(p[2], dtype=np.float32) for p in pages])
            sim = calculate_similarity_matrix(embeddings)
            
            fig = go.Figure(data=go.Heatmap(z=sim, x=[t[:25] for t in titles], y=[t[:25] for t in titles], colorscale='RdYlGn'))
            fig.update_layout(title="Similarity Matrix", height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔝 Most Similar Pairs")
            pairs = []
            for i in range(len(urls)):
                for j in range(i + 1, len(urls)):
                    pairs.append({'Page 1': titles[i][:35], 'Page 2': titles[j][:35], 'Similarity': round(sim[i][j], 3)})
            st.dataframe(pd.DataFrame(pairs).sort_values('Similarity', ascending=False).head(15), use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Grouped Links by Similarity")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            min_sim = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.05)
        with c2:
            cursor.execute("SELECT DISTINCT source_url FROM internal_links")
            sources = ['All'] + [r[0] for r in cursor.fetchall()]
            filter_src = st.selectbox("Source URL", sources)
        with c3:
            cursor.execute("SELECT DISTINCT target_url FROM internal_links")
            targets = ['All'] + [r[0] for r in cursor.fetchall()]
            filter_tgt = st.selectbox("Target URL", targets)
        
        query = '''SELECT il.source_url, il.target_url, il.anchor_text, il.similarity_score,
                   p1.meta_title, p2.meta_title, p1.cluster_name, p2.cluster_name
                   FROM internal_links il
                   LEFT JOIN pages p1 ON il.source_url = p1.url
                   LEFT JOIN pages p2 ON il.target_url = p2.url
                   WHERE il.similarity_score >= ?'''
        params = [min_sim]
        
        if filter_src != 'All':
            query += ' AND il.source_url = ?'
            params.append(filter_src)
        if filter_tgt != 'All':
            query += ' AND il.target_url = ?'
            params.append(filter_tgt)
        
        query += ' ORDER BY il.similarity_score DESC LIMIT 200'
        cursor.execute(query, params)
        links = cursor.fetchall()
        
        if links:
            grouped = {}
            for l in links:
                src = l[0]
                if src not in grouped:
                    grouped[src] = {'title': l[4] or src, 'cluster': l[6], 'links': []}
                grouped[src]['links'].append({'target': l[1], 'title': l[5] or l[1], 'anchor': l[2], 'sim': l[3], 'cluster': l[7]})
            
            for src, data in grouped.items():
                with st.expander(f"📄 {data['title'][:50]}... ({len(data['links'])} links)"):
                    st.markdown(f"**Source:** [{src}]({src})")
                    st.markdown(f"**Cluster:** {data['cluster'] or 'N/A'}")
                    st.divider()
                    
                    for lnk in data['links']:
                        cls = 'high' if lnk['sim'] > 0.7 else ('medium' if lnk['sim'] > 0.5 else 'low')
                        st.markdown(f"""<div class="link-group">
                            <strong>→ {lnk['title'][:45]}...</strong><br>
                            <small>Anchor: "{lnk['anchor'][:35]}..."</small><br>
                            <span class="similarity-{cls}">Similarity: {lnk['sim']:.3f}</span> | Cluster: {lnk['cluster'] or 'N/A'}
                        </div>""", unsafe_allow_html=True)
            
            st.download_button("📥 Export", pd.DataFrame([{'Source': l[0], 'Target': l[1], 'Anchor': l[2], 'Similarity': l[3]} for l in links]).to_csv(index=False), "grouped_links.csv")
        else:
            st.info("No links found. Lower the threshold.")


def show_link_suggestions(conn, ai_provider):
    st.markdown('<p class="main-header">💡 Link Suggestions</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    tab1, tab2 = st.tabs(["🔄 Generate", "📋 View"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Per page", 1, 20, 5)
            min_thresh = st.slider("Min similarity", 0.0, 1.0, 0.3, 0.05)
        with col2:
            active = ai_provider.get_active_provider()
            use_ai = st.checkbox("Use AI", disabled=not active)
            if active:
                st.success(f"AI: {active['provider'].upper()}")
        
        if st.button("🚀 Generate", type="primary"):
            with st.spinner("Generating..."):
                cursor.execute("SELECT url, meta_title, body_content, embedding FROM pages WHERE crawl_status = 'success' AND embedding IS NOT NULL")
                pages = cursor.fetchall()
                
                if len(pages) < 2:
                    st.error("Need 2+ pages with embeddings")
                    return
                
                cursor.execute("SELECT source_url, target_url FROM internal_links")
                existing = set((r[0], r[1]) for r in cursor.fetchall())
                
                urls = [p[0] for p in pages]
                titles = [p[1] for p in pages]
                contents = [p[2] for p in pages]
                embeddings = np.array([np.frombuffer(p[3], dtype=np.float32) for p in pages])
                sim = calculate_similarity_matrix(embeddings)
                
                suggestions = []
                prog = st.progress(0)
                
                for i, src in enumerate(urls):
                    prog.progress((i + 1) / len(urls))
                    indices = np.argsort(sim[i])[::-1]
                    count = 0
                    
                    for j in indices:
                        if count >= top_k or sim[i][j] < min_thresh:
                            break
                        if i == j or (src, urls[j]) in existing:
                            continue
                        
                        priority = 'high' if sim[i][j] >= 0.7 else ('medium' if sim[i][j] >= 0.5 else 'low')
                        anchor = titles[j][:50] if titles[j] else urls[j].split('/')[-1]
                        
                        ai_exp = ''
                        if use_ai and active and count < 2:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                res = loop.run_until_complete(ai_provider.generate_link_suggestion_ai(
                                    contents[i][:1200] if contents[i] else '', contents[j][:1200] if contents[j] else '', active))
                                ai_exp = res.get('explanation', '')
                                if 'anchor_text' in res:
                                    anchor = res['anchor_text']
                            finally:
                                loop.close()
                        
                        suggestions.append({'src': src, 'tgt': urls[j], 'anchor': anchor, 'sim': sim[i][j], 'priority': priority, 'ai': ai_exp})
                        count += 1
                
                cursor.execute('DELETE FROM suggestions')
                for s in suggestions:
                    cursor.execute('INSERT OR REPLACE INTO suggestions (source_url, target_url, suggested_anchor, relevance_score, priority, ai_explanation, status) VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (s['src'], s['tgt'], s['anchor'], s['sim'], s['priority'], s['ai'], 'pending'))
                conn.commit()
                
                st.success(f"✅ Generated {len(suggestions)} suggestions!")
    
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            priority = st.selectbox("Priority", ['All', 'high', 'medium', 'low'])
        with c2:
            status = st.selectbox("Status", ['All', 'pending', 'approved', 'rejected'])
        with c3:
            sort = st.selectbox("Sort", ['Score ↓', 'Score ↑'])
        
        query = 'SELECT * FROM suggestions WHERE 1=1'
        params = []
        if priority != 'All':
            query += ' AND priority = ?'
            params.append(priority)
        if status != 'All':
            query += ' AND status = ?'
            params.append(status)
        query += ' ORDER BY relevance_score ' + ('DESC' if '↓' in sort else 'ASC') + ' LIMIT 100'
        
        cursor.execute(query, params)
        suggestions = cursor.fetchall()
        
        if suggestions:
            for s in suggestions:
                d = dict(s)
                icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(d['priority'], '⚪')
                
                with st.expander(f"{icon} {d['source_url'][:40]}... → {d['target_url'][:40]}..."):
                    st.markdown(f"**Source:** [{d['source_url']}]({d['source_url']})")
                    st.markdown(f"**Target:** [{d['target_url']}]({d['target_url']})")
                    st.markdown(f"**Anchor:** `{d['suggested_anchor']}`")
                    st.markdown(f"**Score:** {d['relevance_score']:.3f} | **Priority:** {d['priority']} | **Status:** {d['status']}")
                    
                    if d['ai_explanation']:
                        st.info(f"**AI:** {d['ai_explanation']}")
                    
                    if d['status'] == 'pending':
                        c1, c2 = st.columns(2)
                        if c1.button("✅ Approve", key=f"a{d['id']}"):
                            cursor.execute('UPDATE suggestions SET status = ? WHERE id = ?', ('approved', d['id']))
                            conn.commit()
                            st.rerun()
                        if c2.button("❌ Reject", key=f"r{d['id']}"):
                            cursor.execute('UPDATE suggestions SET status = ? WHERE id = ?', ('rejected', d['id']))
                            conn.commit()
                            st.rerun()
            
            st.download_button("📥 Export", pd.DataFrame([dict(s) for s in suggestions]).to_csv(index=False), "suggestions.csv")
        else:
            st.info("No suggestions found")


def show_ai_settings(conn, ai_provider):
    st.markdown('<p class="main-header">🔌 AI Settings</p>', unsafe_allow_html=True)
    
    active = ai_provider.get_active_provider()
    if active:
        st.success(f"✅ Active: **{active['provider'].upper()}** ({active['model']})")
    else:
        st.warning("⚠️ No AI configured")
    
    st.divider()
    st.subheader("Configure API Keys")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🤖 OpenAI")
        key = st.text_input("API Key", type="password", key="oai")
        model = st.selectbox("Model", ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'], key="oaim")
        if st.button("Save OpenAI"):
            if key:
                ai_provider.save_api_key('openai', key, model, True)
                st.success("✅ Saved!")
                st.rerun()
    
    with c2:
        st.markdown("### 🧠 Anthropic")
        key = st.text_input("API Key", type="password", key="ant")
        model = st.selectbox("Model", ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'], key="antm")
        if st.button("Save Anthropic"):
            if key:
                ai_provider.save_api_key('anthropic', key, model, True)
                st.success("✅ Saved!")
                st.rerun()
    
    with c3:
        st.markdown("### 💎 Google")
        key = st.text_input("API Key", type="password", key="ggl")
        model = st.selectbox("Model", ['gemini-pro', 'gemini-1.5-pro'], key="gglm")
        if st.button("Save Google"):
            if key:
                ai_provider.save_api_key('google', key, model, True)
                st.success("✅ Saved!")
                st.rerun()
    
    st.divider()
    st.subheader("Saved Keys")
    settings = ai_provider.get_all_api_settings()
    if settings:
        for s in settings:
            d = dict(s)
            status = "🟢 Active" if d['is_active'] else "⚪ Inactive"
            key_preview = d['api_key'][:8] + '...' + d['api_key'][-4:] if d['api_key'] else 'None'
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.write(f"**{d['provider'].upper()}**")
            c2.write(d['model'])
            c3.write(f"`{key_preview}`")
            c4.write(status)
            
            if not d['is_active']:
                if st.button(f"Activate", key=f"act_{d['provider']}"):
                    ai_provider.save_api_key(d['provider'], d['api_key'], d['model'], True)
                    st.rerun()


def show_import_export(conn):
    st.markdown('<p class="main-header">📥 Import/Export</p>', unsafe_allow_html=True)
    
    cursor = conn.cursor()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📤 Export")
        export_type = st.selectbox("Export", ['Pages', 'Links', 'Suggestions', 'All (JSON)'])
        
        if st.button("📥 Export"):
            if export_type == 'Pages':
                cursor.execute("SELECT url, meta_title, word_count, cluster_name FROM pages")
                df = pd.DataFrame(cursor.fetchall(), columns=['URL', 'Title', 'Words', 'Cluster'])
                st.download_button("Download", df.to_csv(index=False), "pages.csv")
            elif export_type == 'Links':
                cursor.execute("SELECT source_url, target_url, anchor_text, similarity_score FROM internal_links")
                df = pd.DataFrame(cursor.fetchall(), columns=['Source', 'Target', 'Anchor', 'Similarity'])
                st.download_button("Download", df.to_csv(index=False), "links.csv")
            elif export_type == 'Suggestions':
                cursor.execute("SELECT source_url, target_url, suggested_anchor, relevance_score, priority, status FROM suggestions")
                df = pd.DataFrame(cursor.fetchall(), columns=['Source', 'Target', 'Anchor', 'Score', 'Priority', 'Status'])
                st.download_button("Download", df.to_csv(index=False), "suggestions.csv")
            else:
                data = {}
                cursor.execute("SELECT url, meta_title, word_count, cluster_name FROM pages")
                data['pages'] = [dict(zip(['url', 'title', 'words', 'cluster'], r)) for r in cursor.fetchall()]
                cursor.execute("SELECT source_url, target_url, anchor_text, similarity_score FROM internal_links")
                data['links'] = [dict(zip(['source', 'target', 'anchor', 'similarity'], r)) for r in cursor.fetchall()]
                st.download_button("Download", json.dumps(data, indent=2), "export.json")
    
    with c2:
        st.subheader("📥 Import")
        uploaded = st.file_uploader("Upload CSV/JSON", type=['csv', 'json'])
        if uploaded:
            if uploaded.name.endswith('.csv'):
                st.dataframe(pd.read_csv(uploaded).head())
            else:
                st.json(json.load(uploaded))
        
        st.divider()
        st.subheader("🗑️ Clear Data")
        if st.button("🗑️ Clear All", type="secondary"):
            if st.checkbox("Confirm delete"):
                cursor.execute('DELETE FROM pages')
                cursor.execute('DELETE FROM internal_links')
                cursor.execute('DELETE FROM suggestions')
                conn.commit()
                st.success("Cleared!")
                st.rerun()


if __name__ == "__main__":
    main()
