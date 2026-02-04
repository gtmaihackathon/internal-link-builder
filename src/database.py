"""
Database Operations Module
Handles all SQLite database interactions for the Internal Link Builder
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import json
import os

# Default database path
DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "data/internal_links.db")


@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH):
    """
    Context manager for database connections.
    
    Usage:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pages")
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Initialize the database with all required tables.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite connection object
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Pages table - stores crawled page data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pages (
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
            images_count INTEGER DEFAULT 0,
            crawl_status TEXT DEFAULT 'pending',
            crawl_error TEXT,
            http_status INTEGER,
            content_type TEXT,
            crawled_at TIMESTAMP,
            updated_at TIMESTAMP,
            embedding BLOB,
            cluster_id INTEGER,
            cluster_name TEXT,
            page_rank REAL DEFAULT 0.0,
            depth INTEGER DEFAULT 0
        )
    ''')
    
    # Internal links table - stores link relationships
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS internal_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url TEXT NOT NULL,
            target_url TEXT NOT NULL,
            anchor_text TEXT,
            context TEXT,
            position INTEGER,
            is_dofollow INTEGER DEFAULT 1,
            is_in_nav INTEGER DEFAULT 0,
            is_in_footer INTEGER DEFAULT 0,
            is_in_content INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_url, target_url, anchor_text)
        )
    ''')
    
    # External links table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS external_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url TEXT NOT NULL,
            target_url TEXT NOT NULL,
            anchor_text TEXT,
            is_dofollow INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_url, target_url, anchor_text)
        )
    ''')
    
    # Link suggestions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url TEXT NOT NULL,
            target_url TEXT NOT NULL,
            suggested_anchor TEXT,
            suggested_context TEXT,
            insertion_point TEXT,
            relevance_score REAL,
            action TEXT DEFAULT 'add',
            priority TEXT DEFAULT 'medium',
            reason TEXT,
            status TEXT DEFAULT 'pending',
            reviewed_by TEXT,
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_url, target_url, action)
        )
    ''')
    
    # Clusters table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            keywords TEXT,
            page_count INTEGER DEFAULT 0,
            pillar_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')
    
    # Crawl jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crawl_jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'queued',
            total_urls INTEGER DEFAULT 0,
            crawled_urls INTEGER DEFAULT 0,
            failed_urls INTEGER DEFAULT 0,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            settings TEXT
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_domain ON pages(domain)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_cluster ON pages(cluster_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(crawl_status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_links_source ON internal_links(source_url)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_links_target ON internal_links(target_url)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status)')
    
    conn.commit()
    return conn


# ============ PAGE OPERATIONS ============

def insert_page(conn: sqlite3.Connection, page_data: Dict[str, Any]) -> int:
    """
    Insert or update a page record.
    
    Args:
        conn: Database connection
        page_data: Dictionary with page data
        
    Returns:
        Row ID of inserted/updated page
    """
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO pages 
        (url, domain, meta_title, meta_description, h1, h2_tags, body_content,
         word_count, internal_links_count, external_links_count, images_count,
         crawl_status, crawl_error, http_status, content_type, crawled_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        page_data.get('url'),
        page_data.get('domain'),
        page_data.get('meta_title'),
        page_data.get('meta_description'),
        page_data.get('h1'),
        json.dumps(page_data.get('h2_tags', [])),
        page_data.get('body_content'),
        page_data.get('word_count', 0),
        page_data.get('internal_links_count', 0),
        page_data.get('external_links_count', 0),
        page_data.get('images_count', 0),
        page_data.get('crawl_status', 'success'),
        page_data.get('crawl_error'),
        page_data.get('http_status'),
        page_data.get('content_type'),
        page_data.get('crawled_at', datetime.now()),
        datetime.now()
    ))
    
    conn.commit()
    return cursor.lastrowid


def get_page(conn: sqlite3.Connection, url: str) -> Optional[Dict[str, Any]]:
    """Get a page by URL."""
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pages WHERE url = ?', (url,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_all_pages(conn: sqlite3.Connection, 
                  status: str = 'success',
                  domain: Optional[str] = None,
                  limit: int = 10000,
                  offset: int = 0) -> List[Dict[str, Any]]:
    """Get all pages with optional filtering."""
    cursor = conn.cursor()
    
    query = 'SELECT * FROM pages WHERE crawl_status = ?'
    params = [status]
    
    if domain:
        query += ' AND domain = ?'
        params.append(domain)
    
    query += ' ORDER BY crawled_at DESC LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_page_count(conn: sqlite3.Connection, status: str = 'success') -> int:
    """Get count of pages by status."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM pages WHERE crawl_status = ?', (status,))
    return cursor.fetchone()['count']


def update_page_embedding(conn: sqlite3.Connection, url: str, embedding: bytes, cluster_id: Optional[int] = None):
    """Update page embedding and optionally cluster ID."""
    cursor = conn.cursor()
    
    if cluster_id is not None:
        cursor.execute(
            'UPDATE pages SET embedding = ?, cluster_id = ?, updated_at = ? WHERE url = ?',
            (embedding, cluster_id, datetime.now(), url)
        )
    else:
        cursor.execute(
            'UPDATE pages SET embedding = ?, updated_at = ? WHERE url = ?',
            (embedding, datetime.now(), url)
        )
    
    conn.commit()


def update_page_cluster(conn: sqlite3.Connection, url: str, cluster_id: int, cluster_name: Optional[str] = None):
    """Update page cluster assignment."""
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE pages SET cluster_id = ?, cluster_name = ?, updated_at = ? WHERE url = ?',
        (cluster_id, cluster_name, datetime.now(), url)
    )
    conn.commit()


# ============ LINK OPERATIONS ============

def insert_internal_link(conn: sqlite3.Connection, link_data: Dict[str, Any]) -> int:
    """Insert an internal link record."""
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO internal_links 
            (source_url, target_url, anchor_text, context, position, 
             is_dofollow, is_in_nav, is_in_footer, is_in_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            link_data.get('source_url'),
            link_data.get('target_url'),
            link_data.get('anchor_text', ''),
            link_data.get('context', ''),
            link_data.get('position', 0),
            link_data.get('is_dofollow', 1),
            link_data.get('is_in_nav', 0),
            link_data.get('is_in_footer', 0),
            link_data.get('is_in_content', 1)
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return 0


def get_internal_links(conn: sqlite3.Connection, 
                       source_url: Optional[str] = None,
                       target_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get internal links with optional filtering."""
    cursor = conn.cursor()
    
    if source_url and target_url:
        cursor.execute(
            'SELECT * FROM internal_links WHERE source_url = ? AND target_url = ?',
            (source_url, target_url)
        )
    elif source_url:
        cursor.execute('SELECT * FROM internal_links WHERE source_url = ?', (source_url,))
    elif target_url:
        cursor.execute('SELECT * FROM internal_links WHERE target_url = ?', (target_url,))
    else:
        cursor.execute('SELECT * FROM internal_links')
    
    return [dict(row) for row in cursor.fetchall()]


def get_link_count(conn: sqlite3.Connection) -> int:
    """Get total count of internal links."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM internal_links')
    return cursor.fetchone()['count']


def get_incoming_links_count(conn: sqlite3.Connection, url: str) -> int:
    """Get count of incoming links to a URL."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM internal_links WHERE target_url = ?', (url,))
    return cursor.fetchone()['count']


def get_outgoing_links_count(conn: sqlite3.Connection, url: str) -> int:
    """Get count of outgoing links from a URL."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM internal_links WHERE source_url = ?', (url,))
    return cursor.fetchone()['count']


# ============ SUGGESTION OPERATIONS ============

def insert_suggestion(conn: sqlite3.Connection, suggestion: Dict[str, Any]) -> int:
    """Insert a link suggestion."""
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO suggestions 
            (source_url, target_url, suggested_anchor, suggested_context,
             insertion_point, relevance_score, action, priority, reason, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            suggestion.get('source_url'),
            suggestion.get('target_url'),
            suggestion.get('suggested_anchor'),
            suggestion.get('suggested_context'),
            suggestion.get('insertion_point'),
            suggestion.get('relevance_score', 0.0),
            suggestion.get('action', 'add'),
            suggestion.get('priority', 'medium'),
            suggestion.get('reason'),
            suggestion.get('status', 'pending')
        ))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return 0


def get_suggestions(conn: sqlite3.Connection,
                    status: str = 'pending',
                    priority: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
    """Get link suggestions with optional filtering."""
    cursor = conn.cursor()
    
    query = 'SELECT * FROM suggestions WHERE status = ?'
    params = [status]
    
    if priority:
        query += ' AND priority = ?'
        params.append(priority)
    
    query += ' ORDER BY relevance_score DESC LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def update_suggestion_status(conn: sqlite3.Connection, 
                             suggestion_id: int, 
                             status: str,
                             reviewed_by: Optional[str] = None):
    """Update suggestion status."""
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE suggestions SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?',
        (status, reviewed_by, datetime.now(), suggestion_id)
    )
    conn.commit()


def clear_suggestions(conn: sqlite3.Connection):
    """Clear all suggestions."""
    cursor = conn.cursor()
    cursor.execute('DELETE FROM suggestions')
    conn.commit()


def get_suggestion_count(conn: sqlite3.Connection, status: str = 'pending') -> int:
    """Get count of suggestions by status."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM suggestions WHERE status = ?', (status,))
    return cursor.fetchone()['count']


# ============ CLUSTER OPERATIONS ============

def insert_cluster(conn: sqlite3.Connection, cluster_data: Dict[str, Any]) -> int:
    """Insert or update a cluster."""
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO clusters 
        (id, name, description, keywords, page_count, pillar_url, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        cluster_data.get('id'),
        cluster_data.get('name'),
        cluster_data.get('description'),
        json.dumps(cluster_data.get('keywords', [])),
        cluster_data.get('page_count', 0),
        cluster_data.get('pillar_url'),
        datetime.now()
    ))
    
    conn.commit()
    return cursor.lastrowid


def get_clusters(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all clusters."""
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clusters ORDER BY page_count DESC')
    return [dict(row) for row in cursor.fetchall()]


def get_cluster_count(conn: sqlite3.Connection) -> int:
    """Get count of unique clusters."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT cluster_id) as count FROM pages WHERE cluster_id IS NOT NULL')
    return cursor.fetchone()['count']


# ============ ANALYSIS QUERIES ============

def get_orphan_pages(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get pages with no incoming internal links."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.url, p.meta_title, p.h1, p.word_count,
               (SELECT COUNT(*) FROM internal_links WHERE source_url = p.url) as outgoing_links
        FROM pages p
        WHERE p.crawl_status = 'success'
        AND p.url NOT IN (SELECT DISTINCT target_url FROM internal_links)
        ORDER BY p.word_count DESC
    ''')
    return [dict(row) for row in cursor.fetchall()]


def get_top_linked_pages(conn: sqlite3.Connection, limit: int = 10) -> List[Dict[str, Any]]:
    """Get pages with most incoming links."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            il.target_url as url,
            p.meta_title as title,
            COUNT(*) as incoming_links
        FROM internal_links il
        LEFT JOIN pages p ON il.target_url = p.url
        GROUP BY il.target_url
        ORDER BY incoming_links DESC
        LIMIT ?
    ''', (limit,))
    return [dict(row) for row in cursor.fetchall()]


def get_pages_by_cluster(conn: sqlite3.Connection, cluster_id: int) -> List[Dict[str, Any]]:
    """Get all pages in a cluster."""
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM pages WHERE cluster_id = ? AND crawl_status = ?',
        (cluster_id, 'success')
    )
    return [dict(row) for row in cursor.fetchall()]


def get_existing_links_set(conn: sqlite3.Connection) -> set:
    """Get set of existing (source, target) link pairs."""
    cursor = conn.cursor()
    cursor.execute('SELECT source_url, target_url FROM internal_links')
    return {(row['source_url'], row['target_url']) for row in cursor.fetchall()}


def get_statistics(conn: sqlite3.Connection) -> Dict[str, int]:
    """Get overall statistics."""
    return {
        'total_pages': get_page_count(conn, 'success'),
        'failed_pages': get_page_count(conn, 'error'),
        'total_links': get_link_count(conn),
        'pending_suggestions': get_suggestion_count(conn, 'pending'),
        'approved_suggestions': get_suggestion_count(conn, 'approved'),
        'clusters': get_cluster_count(conn),
        'orphan_pages': len(get_orphan_pages(conn))
    }


# ============ CLEANUP OPERATIONS ============

def clear_all_data(conn: sqlite3.Connection):
    """Clear all data from all tables."""
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pages')
    cursor.execute('DELETE FROM internal_links')
    cursor.execute('DELETE FROM external_links')
    cursor.execute('DELETE FROM suggestions')
    cursor.execute('DELETE FROM clusters')
    cursor.execute('DELETE FROM crawl_jobs')
    conn.commit()


def clear_pages(conn: sqlite3.Connection):
    """Clear pages and related data."""
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pages')
    cursor.execute('DELETE FROM internal_links')
    cursor.execute('DELETE FROM external_links')
    conn.commit()


def vacuum_database(conn: sqlite3.Connection):
    """Optimize database by vacuuming."""
    conn.execute('VACUUM')
