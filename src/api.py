"""
FastAPI Backend for Internal Link Builder
Provides REST API endpoints for integration with other tools
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import sqlite3
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
import numpy as np
import json

app = FastAPI(
    title="Internal Link Builder API",
    description="Enterprise-grade internal linking API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db():
    conn = sqlite3.connect('internal_links.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Pydantic models
class CrawlRequest(BaseModel):
    urls: List[HttpUrl]
    user_agent: str = "googlebot"
    max_concurrent: int = 10
    timeout: int = 30
    delay: float = 0.5

class CrawlStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    urls_crawled: int
    urls_total: int
    errors: int

class PageResponse(BaseModel):
    url: str
    meta_title: Optional[str]
    meta_description: Optional[str]
    h1: Optional[str]
    word_count: int
    internal_links_count: int
    external_links_count: int
    cluster_id: Optional[int]

class LinkSuggestion(BaseModel):
    source_url: str
    target_url: str
    suggested_anchor: str
    relevance_score: float
    priority: str
    reason: str

class ClusterRequest(BaseModel):
    method: str = "hdbscan"
    n_clusters: Optional[int] = None
    min_cluster_size: int = 5

# User agents
USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# In-memory job tracking
crawl_jobs = {}

@app.get("/")
async def root():
    return {"message": "Internal Link Builder API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ============ CRAWLING ENDPOINTS ============

@app.post("/api/crawl", response_model=dict)
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start a new crawl job"""
    import uuid
    job_id = str(uuid.uuid4())
    
    crawl_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "urls_crawled": 0,
        "urls_total": len(request.urls),
        "errors": 0
    }
    
    background_tasks.add_task(
        run_crawl_job, 
        job_id, 
        [str(url) for url in request.urls],
        request.user_agent,
        request.max_concurrent,
        request.timeout,
        request.delay
    )
    
    return {"job_id": job_id, "message": "Crawl job started"}

async def run_crawl_job(job_id: str, urls: List[str], user_agent: str, 
                        max_concurrent: int, timeout: int, delay: float):
    """Background task to run crawl"""
    conn = get_db()
    crawl_jobs[job_id]["status"] = "running"
    
    ua_string = USER_AGENTS.get(user_agent, user_agent)
    headers = {"User-Agent": ua_string}
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_url(session, url):
        async with semaphore:
            await asyncio.sleep(delay)
            try:
                async with session.get(url, timeout=ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        html = await response.text()
                        return {"url": url, "html": html, "status": "success"}
                    return {"url": url, "status": "error", "error": f"HTTP {response.status}"}
            except Exception as e:
                return {"url": url, "status": "error", "error": str(e)}
    
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [fetch_url(session, url) for url in urls]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            
            # Process result
            if result["status"] == "success":
                # Parse and save
                parsed = parse_html(result["html"], result["url"])
                save_page(conn, result["url"], parsed)
                crawl_jobs[job_id]["urls_crawled"] += 1
            else:
                crawl_jobs[job_id]["errors"] += 1
            
            crawl_jobs[job_id]["progress"] = (i + 1) / len(urls)
    
    crawl_jobs[job_id]["status"] = "completed"
    conn.close()

def parse_html(html: str, url: str) -> dict:
    """Parse HTML and extract data"""
    soup = BeautifulSoup(html, 'html.parser')
    
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    meta_title = soup.find('title')
    meta_title = meta_title.get_text(strip=True) if meta_title else ''
    
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_desc.get('content', '') if meta_desc else ''
    
    h1_tag = soup.find('h1')
    h1 = h1_tag.get_text(strip=True) if h1_tag else ''
    
    body_content = ''
    main = soup.find('main') or soup.find('article') or soup.find('body')
    if main:
        body_content = main.get_text(separator=' ', strip=True)[:10000]
    
    word_count = len(body_content.split())
    
    # Extract links
    domain = urlparse(url).netloc
    internal_links = []
    external_links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        anchor = a_tag.get_text(strip=True)
        full_url = urljoin(url, href)
        parsed = urlparse(full_url)
        
        if parsed.netloc == domain:
            internal_links.append({'target_url': full_url, 'anchor_text': anchor})
        elif parsed.scheme in ['http', 'https']:
            external_links.append({'target_url': full_url, 'anchor_text': anchor})
    
    return {
        'meta_title': meta_title,
        'meta_description': meta_description,
        'h1': h1,
        'body_content': body_content,
        'word_count': word_count,
        'internal_links': internal_links,
        'external_links': external_links
    }

def save_page(conn, url: str, parsed: dict):
    """Save page to database"""
    c = conn.cursor()
    domain = urlparse(url).netloc
    
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
    
    for link in parsed['internal_links']:
        c.execute("""
            INSERT OR IGNORE INTO internal_links 
            (source_url, target_url, anchor_text)
            VALUES (?, ?, ?)
        """, (url, link['target_url'], link['anchor_text']))
    
    conn.commit()

@app.get("/api/crawl/{job_id}", response_model=CrawlStatus)
async def get_crawl_status(job_id: str):
    """Get status of a crawl job"""
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = crawl_jobs[job_id]
    return CrawlStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        urls_crawled=job["urls_crawled"],
        urls_total=job["urls_total"],
        errors=job["errors"]
    )

# ============ PAGES ENDPOINTS ============

@app.get("/api/pages", response_model=List[PageResponse])
async def get_pages(
    domain: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get crawled pages"""
    conn = get_db()
    
    query = """
        SELECT url, meta_title, meta_description, h1, word_count,
               internal_links_count, external_links_count, cluster_id
        FROM pages
        WHERE crawl_status = 'success'
    """
    params = []
    
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    
    query += " ORDER BY crawled_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    return [PageResponse(**dict(row)) for row in rows]

@app.get("/api/pages/{url:path}", response_model=dict)
async def get_page_detail(url: str):
    """Get detailed page information"""
    conn = get_db()
    
    page = conn.execute(
        "SELECT * FROM pages WHERE url = ?", (url,)
    ).fetchone()
    
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Get internal links
    outgoing = conn.execute(
        "SELECT target_url, anchor_text FROM internal_links WHERE source_url = ?",
        (url,)
    ).fetchall()
    
    incoming = conn.execute(
        "SELECT source_url, anchor_text FROM internal_links WHERE target_url = ?",
        (url,)
    ).fetchall()
    
    conn.close()
    
    return {
        "page": dict(page),
        "outgoing_links": [dict(r) for r in outgoing],
        "incoming_links": [dict(r) for r in incoming]
    }

# ============ ANALYSIS ENDPOINTS ============

@app.post("/api/cluster")
async def generate_clusters(request: ClusterRequest, background_tasks: BackgroundTasks):
    """Generate semantic clusters"""
    background_tasks.add_task(run_clustering, request.method, request.n_clusters, request.min_cluster_size)
    return {"message": "Clustering job started"}

async def run_clustering(method: str, n_clusters: Optional[int], min_cluster_size: int):
    """Background clustering task"""
    from sentence_transformers import SentenceTransformer
    import hdbscan
    from sklearn.cluster import KMeans
    
    conn = get_db()
    pages = conn.execute("""
        SELECT url, meta_title, h1, body_content
        FROM pages WHERE crawl_status = 'success'
    """).fetchall()
    
    if len(pages) < 5:
        return
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{p['meta_title']} {p['h1']} {p['body_content'][:1000]}" for p in pages]
    embeddings = model.encode(texts)
    
    # Cluster
    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)
    else:
        if n_clusters is None:
            n_clusters = min(20, len(pages) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
    
    # Save results
    c = conn.cursor()
    for i, page in enumerate(pages):
        embedding_bytes = embeddings[i].tobytes()
        c.execute(
            "UPDATE pages SET embedding = ?, cluster_id = ? WHERE url = ?",
            (embedding_bytes, int(labels[i]), page['url'])
        )
    
    conn.commit()
    conn.close()

@app.get("/api/suggestions", response_model=List[LinkSuggestion])
async def get_suggestions(
    priority: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get link suggestions"""
    conn = get_db()
    
    query = "SELECT * FROM suggestions WHERE status = 'pending'"
    params = []
    
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    
    query += " ORDER BY relevance_score DESC LIMIT ?"
    params.append(limit)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    return [LinkSuggestion(**dict(row)) for row in rows]

@app.post("/api/suggestions/generate")
async def generate_suggestions(background_tasks: BackgroundTasks, top_k: int = 5):
    """Generate link suggestions based on semantic similarity"""
    background_tasks.add_task(run_suggestion_generation, top_k)
    return {"message": "Suggestion generation started"}

async def run_suggestion_generation(top_k: int):
    """Background suggestion generation"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    conn = get_db()
    pages = conn.execute("""
        SELECT url, meta_title, h1, embedding
        FROM pages WHERE embedding IS NOT NULL
    """).fetchall()
    
    if len(pages) < 2:
        return
    
    # Load embeddings
    urls = [p['url'] for p in pages]
    embeddings = np.array([np.frombuffer(p['embedding'], dtype=np.float32) for p in pages])
    
    # Calculate similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Get existing links
    existing = set()
    for row in conn.execute("SELECT source_url, target_url FROM internal_links"):
        existing.add((row['source_url'], row['target_url']))
    
    # Generate suggestions
    c = conn.cursor()
    c.execute("DELETE FROM suggestions")
    
    for i, source_url in enumerate(urls):
        top_indices = np.argsort(sim_matrix[i])[::-1][1:top_k+1]
        
        for j in top_indices:
            target_url = urls[j]
            
            if (source_url, target_url) in existing:
                continue
            
            similarity = sim_matrix[i][j]
            if similarity < 0.3:
                continue
            
            target_page = pages[j]
            anchor = target_page['h1'] or target_page['meta_title'] or target_url.split('/')[-1]
            
            priority = 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
            
            c.execute("""
                INSERT INTO suggestions 
                (source_url, target_url, suggested_anchor, relevance_score, action, priority, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                source_url, target_url, anchor[:100], float(similarity),
                'add', priority, f'Semantic similarity: {similarity:.2f}'
            ))
    
    conn.commit()
    conn.close()

@app.get("/api/orphans")
async def get_orphan_pages():
    """Get pages with no incoming internal links"""
    conn = get_db()
    
    orphans = conn.execute("""
        SELECT p.url, p.meta_title, p.word_count
        FROM pages p
        WHERE p.crawl_status = 'success'
        AND p.url NOT IN (SELECT DISTINCT target_url FROM internal_links)
    """).fetchall()
    
    conn.close()
    
    return [dict(row) for row in orphans]

@app.get("/api/cannibalization")
async def get_cannibalization(threshold: float = 0.85):
    """Get potentially cannibalizing pages"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    conn = get_db()
    pages = conn.execute("""
        SELECT url, meta_title, embedding
        FROM pages WHERE embedding IS NOT NULL
    """).fetchall()
    
    if len(pages) < 2:
        return []
    
    embeddings = np.array([np.frombuffer(p['embedding'], dtype=np.float32) for p in pages])
    sim_matrix = cosine_similarity(embeddings)
    
    results = []
    for i in range(len(pages)):
        for j in range(i+1, len(pages)):
            if sim_matrix[i][j] > threshold:
                results.append({
                    "url1": pages[i]['url'],
                    "url2": pages[j]['url'],
                    "title1": pages[i]['meta_title'],
                    "title2": pages[j]['meta_title'],
                    "similarity": float(sim_matrix[i][j])
                })
    
    conn.close()
    return results

# ============ EXPORT ENDPOINTS ============

@app.get("/api/export/pages")
async def export_pages(format: str = "json"):
    """Export all pages data"""
    conn = get_db()
    pages = conn.execute("SELECT * FROM pages").fetchall()
    conn.close()
    
    data = [dict(row) for row in pages]
    
    if format == "csv":
        import csv
        from io import StringIO
        output = StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return {"data": output.getvalue(), "format": "csv"}
    
    return {"data": data, "format": "json"}

@app.get("/api/export/links")
async def export_links(format: str = "json"):
    """Export all internal links"""
    conn = get_db()
    links = conn.execute("SELECT * FROM internal_links").fetchall()
    conn.close()
    
    data = [dict(row) for row in links]
    return {"data": data, "format": format}

@app.get("/api/stats")
async def get_stats():
    """Get overview statistics"""
    conn = get_db()
    
    stats = {
        "total_pages": conn.execute("SELECT COUNT(*) FROM pages WHERE crawl_status='success'").fetchone()[0],
        "total_links": conn.execute("SELECT COUNT(*) FROM internal_links").fetchone()[0],
        "pending_suggestions": conn.execute("SELECT COUNT(*) FROM suggestions WHERE status='pending'").fetchone()[0],
        "clusters": conn.execute("SELECT COUNT(DISTINCT cluster_id) FROM pages WHERE cluster_id IS NOT NULL").fetchone()[0],
        "orphan_pages": conn.execute("""
            SELECT COUNT(*) FROM pages p 
            WHERE crawl_status='success' 
            AND url NOT IN (SELECT DISTINCT target_url FROM internal_links)
        """).fetchone()[0]
    }
    
    conn.close()
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
