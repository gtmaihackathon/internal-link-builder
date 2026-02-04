#!/usr/bin/env python3
"""
Internal Link Builder CLI
Command-line interface for batch processing and automation
"""

import argparse
import asyncio
import aiohttp
from aiohttp import ClientTimeout
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import json
import sys
from tqdm import tqdm
from pathlib import Path

# User agents
USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_mobile': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def init_db(db_path='internal_links.db'):
    """Initialize SQLite database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
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
    
    c.execute('''CREATE TABLE IF NOT EXISTS internal_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT,
        target_url TEXT,
        anchor_text TEXT,
        context TEXT,
        is_dofollow INTEGER DEFAULT 1,
        UNIQUE(source_url, target_url, anchor_text)
    )''')
    
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
    
    conn.commit()
    return conn

async def crawl_url(session, url, semaphore, delay, timeout):
    """Crawl a single URL"""
    async with semaphore:
        await asyncio.sleep(delay)
        try:
            async with session.get(url, timeout=ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    html = await response.text()
                    return {'url': url, 'html': html, 'status': 'success'}
                else:
                    return {'url': url, 'status': 'error', 'error': f'HTTP {response.status}'}
        except asyncio.TimeoutError:
            return {'url': url, 'status': 'error', 'error': 'Timeout'}
        except Exception as e:
            return {'url': url, 'status': 'error', 'error': str(e)}

def parse_html(html, url):
    """Parse HTML and extract data"""
    soup = BeautifulSoup(html, 'html.parser')
    
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()
    
    meta_title = ''
    title_tag = soup.find('title')
    if title_tag:
        meta_title = title_tag.get_text(strip=True)
    
    meta_description = ''
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        meta_description = meta_desc.get('content', '')
    
    h1 = ''
    h1_tag = soup.find('h1')
    if h1_tag:
        h1 = h1_tag.get_text(strip=True)
    
    body_content = ''
    main = soup.find('main') or soup.find('article') or soup.find('body')
    if main:
        body_content = main.get_text(separator=' ', strip=True)[:10000]
    
    word_count = len(body_content.split())
    
    domain = urlparse(url).netloc
    internal_links = []
    external_links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        anchor_text = a_tag.get_text(strip=True)
        full_url = urljoin(url, href)
        parsed = urlparse(full_url)
        
        if parsed.netloc == domain:
            if not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                internal_links.append({
                    'target_url': full_url,
                    'anchor_text': anchor_text
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

async def crawl_urls(urls, user_agent, max_concurrent, timeout, delay):
    """Crawl multiple URLs"""
    headers = {'User-Agent': USER_AGENTS.get(user_agent, user_agent)}
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=5)
    
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [crawl_url(session, url, semaphore, delay, timeout) for url in urls]
        
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Crawling"):
            result = await coro
            results.append(result)
        
        return results

def generate_embeddings(texts, batch_size=32):
    """Generate embeddings"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def cluster_pages(embeddings, method='hdbscan', n_clusters=None, min_cluster_size=5):
    """Cluster pages"""
    if method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)
    else:
        from sklearn.cluster import KMeans
        if n_clusters is None:
            n_clusters = min(20, len(embeddings) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
    
    return labels

def generate_suggestions(conn, top_k=5):
    """Generate link suggestions"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Load pages with embeddings
    pages = pd.read_sql("""
        SELECT url, meta_title, h1, embedding
        FROM pages WHERE embedding IS NOT NULL
    """, conn)
    
    if len(pages) < 2:
        print("Not enough pages with embeddings")
        return []
    
    # Convert embeddings
    embeddings = np.array([
        np.frombuffer(row['embedding'], dtype=np.float32)
        for _, row in pages.iterrows()
    ])
    
    # Calculate similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Get existing links
    existing_links = pd.read_sql("SELECT source_url, target_url FROM internal_links", conn)
    existing_set = set(zip(existing_links['source_url'], existing_links['target_url']))
    
    # Generate suggestions
    suggestions = []
    for i in tqdm(range(len(pages)), desc="Generating suggestions"):
        source_url = pages.iloc[i]['url']
        top_indices = np.argsort(sim_matrix[i])[::-1][1:top_k+1]
        
        for j in top_indices:
            target_url = pages.iloc[j]['url']
            
            if (source_url, target_url) in existing_set:
                continue
            
            similarity = sim_matrix[i][j]
            if similarity < 0.3:
                continue
            
            target_row = pages.iloc[j]
            anchor = target_row['h1'] or target_row['meta_title'] or target_url.split('/')[-1]
            
            priority = 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
            
            suggestions.append({
                'source_url': source_url,
                'target_url': target_url,
                'suggested_anchor': anchor[:100],
                'relevance_score': float(similarity),
                'action': 'add',
                'priority': priority,
                'reason': f'Semantic similarity: {similarity:.2f}'
            })
    
    return suggestions

def cmd_crawl(args):
    """Handle crawl command"""
    conn = init_db(args.db)
    
    # Load URLs
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
        urls = df[args.url_column].dropna().tolist()
    else:
        with open(args.input, 'r') as f:
            urls = [line.strip() for line in f if line.strip().startswith('http')]
    
    print(f"🕷️ Crawling {len(urls):,} URLs...")
    print(f"   User Agent: {args.user_agent}")
    print(f"   Concurrent: {args.concurrent}")
    print(f"   Delay: {args.delay}s")
    
    # Run crawl
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(
            crawl_urls(urls, args.user_agent, args.concurrent, args.timeout, args.delay)
        )
    finally:
        loop.close()
    
    # Process results
    success_count = 0
    error_count = 0
    c = conn.cursor()
    
    for result in tqdm(results, desc="Processing"):
        url = result['url']
        domain = urlparse(url).netloc
        
        if result['status'] == 'success':
            try:
                parsed = parse_html(result['html'], url)
                
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
                    try:
                        c.execute("""
                            INSERT OR IGNORE INTO internal_links 
                            (source_url, target_url, anchor_text)
                            VALUES (?, ?, ?)
                        """, (url, link['target_url'], link['anchor_text']))
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
    conn.close()
    
    print(f"\n✅ Crawl complete!")
    print(f"   Success: {success_count:,}")
    print(f"   Errors: {error_count:,}")

def cmd_embed(args):
    """Handle embed command"""
    conn = init_db(args.db)
    
    print("🔮 Generating embeddings...")
    
    pages = pd.read_sql("""
        SELECT url, meta_title, h1, body_content
        FROM pages WHERE crawl_status = 'success'
    """, conn)
    
    if len(pages) < 2:
        print("❌ Not enough pages to generate embeddings")
        return
    
    texts = []
    for _, row in pages.iterrows():
        text = f"{row['meta_title'] or ''} {row['h1'] or ''} {row['body_content'] or ''}"
        texts.append(text[:2000])
    
    embeddings = generate_embeddings(texts, batch_size=args.batch_size)
    
    # Save embeddings
    c = conn.cursor()
    for i, row in tqdm(pages.iterrows(), total=len(pages), desc="Saving"):
        embedding_bytes = embeddings[i].tobytes()
        c.execute("UPDATE pages SET embedding = ? WHERE url = ?", (embedding_bytes, row['url']))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Generated embeddings for {len(pages):,} pages")

def cmd_cluster(args):
    """Handle cluster command"""
    conn = init_db(args.db)
    
    print(f"🧩 Clustering pages using {args.method}...")
    
    pages = pd.read_sql("""
        SELECT url, embedding FROM pages WHERE embedding IS NOT NULL
    """, conn)
    
    if len(pages) < 5:
        print("❌ Not enough pages with embeddings")
        return
    
    embeddings = np.array([
        np.frombuffer(row['embedding'], dtype=np.float32)
        for _, row in pages.iterrows()
    ])
    
    labels = cluster_pages(
        embeddings, 
        method=args.method, 
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size
    )
    
    # Save cluster labels
    c = conn.cursor()
    for i, row in pages.iterrows():
        c.execute("UPDATE pages SET cluster_id = ? WHERE url = ?", (int(labels[i]), row['url']))
    
    conn.commit()
    
    # Print cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n✅ Created {len(unique):,} clusters")
    print("\nCluster distribution:")
    for cluster_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"   {name}: {count:,} pages")
    
    conn.close()

def cmd_suggest(args):
    """Handle suggest command"""
    conn = init_db(args.db)
    
    print("💡 Generating link suggestions...")
    
    suggestions = generate_suggestions(conn, top_k=args.top_k)
    
    # Save suggestions
    c = conn.cursor()
    c.execute("DELETE FROM suggestions")
    
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
    
    # Export if output specified
    if args.output:
        df = pd.DataFrame(suggestions)
        if args.output.endswith('.json'):
            df.to_json(args.output, orient='records', indent=2)
        else:
            df.to_csv(args.output, index=False)
        print(f"📁 Exported to {args.output}")
    
    conn.close()
    
    print(f"✅ Generated {len(suggestions):,} suggestions")
    
    # Summary
    high = sum(1 for s in suggestions if s['priority'] == 'high')
    medium = sum(1 for s in suggestions if s['priority'] == 'medium')
    low = sum(1 for s in suggestions if s['priority'] == 'low')
    print(f"   🔴 High: {high:,}")
    print(f"   🟡 Medium: {medium:,}")
    print(f"   🟢 Low: {low:,}")

def cmd_export(args):
    """Handle export command"""
    conn = init_db(args.db)
    
    print(f"📤 Exporting data to {args.output}...")
    
    if args.type == 'pages':
        df = pd.read_sql("SELECT * FROM pages", conn)
    elif args.type == 'links':
        df = pd.read_sql("SELECT * FROM internal_links", conn)
    elif args.type == 'suggestions':
        df = pd.read_sql("SELECT * FROM suggestions", conn)
    else:  # all
        data = {
            'pages': pd.read_sql("SELECT * FROM pages", conn).to_dict(orient='records'),
            'internal_links': pd.read_sql("SELECT * FROM internal_links", conn).to_dict(orient='records'),
            'suggestions': pd.read_sql("SELECT * FROM suggestions", conn).to_dict(orient='records')
        }
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ Exported all data to {args.output}")
        conn.close()
        return
    
    if args.output.endswith('.json'):
        df.to_json(args.output, orient='records', indent=2)
    else:
        df.to_csv(args.output, index=False)
    
    conn.close()
    print(f"✅ Exported {len(df):,} records")

def cmd_stats(args):
    """Handle stats command"""
    conn = init_db(args.db)
    
    pages_count = pd.read_sql("SELECT COUNT(*) as c FROM pages WHERE crawl_status='success'", conn).iloc[0]['c']
    links_count = pd.read_sql("SELECT COUNT(*) as c FROM internal_links", conn).iloc[0]['c']
    suggestions_count = pd.read_sql("SELECT COUNT(*) as c FROM suggestions", conn).iloc[0]['c']
    clusters_count = pd.read_sql("SELECT COUNT(DISTINCT cluster_id) as c FROM pages WHERE cluster_id IS NOT NULL", conn).iloc[0]['c']
    orphans_count = pd.read_sql("""
        SELECT COUNT(*) as c FROM pages p
        WHERE crawl_status='success'
        AND url NOT IN (SELECT DISTINCT target_url FROM internal_links)
    """, conn).iloc[0]['c']
    
    print("\n📊 Internal Link Builder Statistics")
    print("=" * 40)
    print(f"📄 Pages crawled:      {pages_count:,}")
    print(f"🔗 Internal links:     {links_count:,}")
    print(f"💡 Suggestions:        {suggestions_count:,}")
    print(f"🧩 Clusters:           {clusters_count:,}")
    print(f"🔍 Orphan pages:       {orphans_count:,}")
    
    conn.close()

def main():
    parser = argparse.ArgumentParser(
        description='Internal Link Builder CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl URLs from file
  python cli.py crawl urls.txt --user-agent googlebot --concurrent 20
  
  # Generate embeddings
  python cli.py embed
  
  # Cluster pages
  python cli.py cluster --method hdbscan --min-cluster-size 5
  
  # Generate suggestions
  python cli.py suggest --top-k 10 --output suggestions.csv
  
  # Export data
  python cli.py export --type all --output export.json
  
  # View stats
  python cli.py stats
        """
    )
    
    parser.add_argument('--db', default='internal_links.db', help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl URLs')
    crawl_parser.add_argument('input', help='Input file (txt or csv)')
    crawl_parser.add_argument('--url-column', default='url', help='URL column name for CSV')
    crawl_parser.add_argument('--user-agent', default='googlebot', choices=list(USER_AGENTS.keys()))
    crawl_parser.add_argument('--concurrent', type=int, default=10, help='Concurrent requests')
    crawl_parser.add_argument('--timeout', type=int, default=30, help='Request timeout')
    crawl_parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings')
    embed_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Cluster pages')
    cluster_parser.add_argument('--method', default='hdbscan', choices=['hdbscan', 'kmeans'])
    cluster_parser.add_argument('--n-clusters', type=int, help='Number of clusters (kmeans)')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=5, help='Min cluster size (hdbscan)')
    
    # Suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Generate suggestions')
    suggest_parser.add_argument('--top-k', type=int, default=5, help='Suggestions per page')
    suggest_parser.add_argument('--output', help='Output file (csv or json)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('--type', default='all', choices=['pages', 'links', 'suggestions', 'all'])
    export_parser.add_argument('--output', required=True, help='Output file')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    if args.command == 'crawl':
        cmd_crawl(args)
    elif args.command == 'embed':
        cmd_embed(args)
    elif args.command == 'cluster':
        cmd_cluster(args)
    elif args.command == 'suggest':
        cmd_suggest(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
