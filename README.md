# 🔗 Internal Link Builder Pro

An enterprise-grade, AI-powered internal linking tool designed to scale to 30-50k+ pages. Built with Python, Streamlit, and modern ML techniques.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

### Core Features
- **Async Web Crawling**: High-performance async crawler using aiohttp
- **Multiple User Agents**: Googlebot, Bingbot, Chrome, or custom UA to bypass blocks
- **Scalable Architecture**: Handle 30-50k+ pages with SQLite storage
- **Rate Limiting**: Configurable delays and concurrent request limits

### Data Extraction
- Meta title, description, H1 tags
- Body content (cleaned, without nav/footer)
- Word count
- All internal links with anchor text and context
- External links tracking

### AI-Powered Analysis
- **Semantic Embeddings**: Using sentence-transformers (all-MiniLM-L6-v2)
- **Topic Clustering**: HDBSCAN or K-means clustering
- **Similarity Analysis**: Cosine similarity for finding related pages
- **Visualization**: UMAP/t-SNE 2D scatter plots

### Link Suggestions
- AI-generated internal link suggestions
- Suggested anchor text based on target page content
- Relevance scores (0-1)
- Priority levels (high/medium/low)
- Orphan page detection
- Content cannibalization detection

## 📦 Installation

### Option 1: Local Installation

```bash
# Clone or download the project
cd internal_link_builder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: Docker

```bash
# Build the image
docker build -t internal-link-builder .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data internal-link-builder
```

### Option 3: Docker Compose

```yaml
version: '3.8'
services:
  internal-link-builder:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## 🎯 Usage Guide

### 1. Crawl URLs

1. Go to **🕷️ Crawl URLs** section
2. Enter URLs (one per line) or upload a CSV/TXT file
3. Configure crawler settings in the sidebar:
   - **User Agent**: Googlebot recommended for bypassing blocks
   - **Concurrent Requests**: 10-50 depending on server capacity
   - **Delay**: 0.5-2 seconds between requests
   - **Timeout**: 30-60 seconds
4. Click **Start Crawling**

### 2. Generate Embeddings & Clusters

1. Go to **🧩 Clusters & Embeddings** section
2. Choose clustering method:
   - **HDBSCAN**: Better for varying cluster sizes, detects noise
   - **K-means**: Fixed number of clusters, faster
3. Choose dimension reduction:
   - **UMAP**: Better preserves global structure
   - **t-SNE**: Better for local relationships
4. Click **Generate Embeddings & Clusters**
5. Explore the interactive 2D visualization

### 3. Get Link Suggestions

1. Go to **💡 Link Suggestions** section
2. Ensure embeddings are generated
3. Set number of suggestions per page (5-20)
4. Click **Generate Link Suggestions**
5. Filter by priority and export to CSV

### 4. Find Issues

- **🔍 Orphan Pages**: Pages with no incoming internal links
- **⚠️ Cannibalization**: Pages competing for same keywords (>85% similarity)

## ⚙️ Configuration Options

### Crawler Settings

| Setting | Default | Description |
|---------|---------|-------------|
| User Agent | Googlebot | Bot identifier sent to servers |
| Concurrent Requests | 10 | Max simultaneous connections |
| Request Delay | 0.5s | Delay between requests (rate limiting) |
| Timeout | 30s | Max wait time per request |

### User Agents Available

- `googlebot`: Standard Googlebot desktop crawler
- `googlebot_mobile`: Googlebot mobile crawler
- `bingbot`: Microsoft Bing crawler
- `chrome`: Chrome browser (for sites blocking bots)
- `custom`: Enter your own user agent string

### Clustering Settings

| Setting | Options | Description |
|---------|---------|-------------|
| Method | HDBSCAN, K-means | Clustering algorithm |
| Min Cluster Size | 2-20 | Minimum pages per cluster (HDBSCAN) |
| N Clusters | 2-50 | Number of clusters (K-means) |
| Dimension Reduction | UMAP, t-SNE | Method for 2D visualization |

## 📊 Output Formats

### Suggestions Export (CSV)
```csv
source_url,target_url,suggested_anchor,relevance_score,priority,reason
https://example.com/page-1,https://example.com/page-2,"Guide to X",0.85,high,"Semantic similarity: 0.85"
```

### Full Export (JSON)
```json
{
  "pages": [...],
  "internal_links": [...],
  "suggestions": [...],
  "exported_at": "2024-01-15T10:30:00"
}
```

## 🏗️ Architecture

```
internal_link_builder/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-container setup
├── README.md           # This file
└── data/
    └── internal_links.db  # SQLite database
```

### Database Schema

**pages**
- url, domain, meta_title, meta_description, h1
- body_content, word_count
- internal_links_count, external_links_count
- crawl_status, crawl_error, crawled_at
- embedding (BLOB), cluster_id, page_rank

**internal_links**
- source_url, target_url, anchor_text
- context, is_dofollow

**suggestions**
- source_url, target_url, suggested_anchor
- relevance_score, action, priority, reason, status

## 🔧 Advanced Features

### PageRank Calculation
The tool includes a built-in PageRank algorithm to identify your most authoritative pages based on internal link structure.

### Content Cannibalization Detection
Automatically finds pages with >85% semantic similarity that may be competing for the same keywords. Adjust the threshold as needed.

### Orphan Page Detection
Identifies pages with zero incoming internal links that may be "orphaned" and hard for search engines to discover.

## 🚀 Scaling to 50k+ Pages

For large-scale crawling:

1. **Increase concurrent requests** to 30-50
2. **Use PostgreSQL** instead of SQLite (modify connection string)
3. **Deploy on cloud** (AWS, GCP, Azure) with more RAM
4. **Consider batch processing** - crawl in batches of 5k-10k
5. **Use Redis** for distributed task queuing

### PostgreSQL Setup (Optional)

```python
# Replace SQLite connection with:
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="internal_links",
    user="your_user",
    password="your_password"
)
```

## 📈 Roadmap / Future Features

- [ ] Google Search Console integration
- [ ] Scheduled automated crawling
- [ ] API endpoints for external integrations
- [ ] WordPress/CMS direct integration
- [ ] Bulk link insertion suggestions
- [ ] Historical tracking and comparison
- [ ] Custom embedding models
- [ ] Multi-language support
- [ ] Link velocity tracking
- [ ] Anchor text diversity analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this tool for personal or commercial purposes.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering
- [UMAP](https://umap-learn.readthedocs.io/) for dimension reduction
- [Streamlit](https://streamlit.io/) for the UI framework
- [aiohttp](https://docs.aiohttp.org/) for async HTTP

---

Built with ❤️ for SEO professionals who want to scale their internal linking
