# 🚀 Complete Setup Guide: Internal Link Builder

A step-by-step guide to setting up, running, and deploying the Internal Link Builder tool.

---

## 📁 GitHub Repository Structure

```
internal-link-builder/
│
├── 📄 README.md                 # Project overview & quick start
├── 📄 SETUP_GUIDE.md            # This file - detailed setup instructions
├── 📄 FEATURES.md               # Future features & roadmap
├── 📄 LICENSE                   # MIT License
├── 📄 .gitignore                # Git ignore rules
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 Dockerfile                # Docker container config
├── 📄 docker-compose.yml        # Docker Compose config
│
├── 📂 src/                      # Source code
│   ├── 📄 __init__.py
│   ├── 📄 app.py                # Streamlit application
│   ├── 📄 api.py                # FastAPI backend
│   ├── 📄 cli.py                # Command-line interface
│   ├── 📄 crawler.py            # Async web crawler
│   ├── 📄 parser.py             # HTML parser
│   ├── 📄 embeddings.py         # Embedding generation
│   ├── 📄 clustering.py         # Clustering algorithms
│   ├── 📄 suggestions.py        # Link suggestion engine
│   └── 📄 database.py           # Database operations
│
├── 📂 config/                   # Configuration files
│   ├── 📄 settings.py           # App settings
│   └── 📄 user_agents.py        # User agent strings
│
├── 📂 tests/                    # Unit tests
│   ├── 📄 __init__.py
│   ├── 📄 test_crawler.py
│   ├── 📄 test_parser.py
│   └── 📄 test_embeddings.py
│
├── 📂 data/                     # Data directory (gitignored)
│   └── 📄 .gitkeep
│
├── 📂 exports/                  # Export directory (gitignored)
│   └── 📄 .gitkeep
│
└── 📂 .github/                  # GitHub specific files
    └── 📂 workflows/
        └── 📄 ci.yml            # GitHub Actions CI/CD
```

---

## 📋 Step 1: Create GitHub Repository

### Option A: Create via GitHub Website

1. Go to [github.com/new](https://github.com/new)
2. Fill in:
   - **Repository name**: `internal-link-builder`
   - **Description**: `AI-powered internal linking tool for SEO at scale`
   - **Visibility**: Public or Private
   - **Initialize**: Check "Add a README file"
   - **Add .gitignore**: Select "Python"
   - **License**: MIT License
3. Click **Create repository**

### Option B: Create via GitHub CLI

```bash
# Install GitHub CLI if not installed
# macOS: brew install gh
# Windows: winget install GitHub.cli
# Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md

# Authenticate
gh auth login

# Create repository
gh repo create internal-link-builder --public --description "AI-powered internal linking tool for SEO at scale"
```

---

## 📋 Step 2: Clone & Setup Local Environment

### 2.1 Clone the Repository

```bash
# Clone your new repository
git clone https://github.com/YOUR_USERNAME/internal-link-builder.git
cd internal-link-builder
```

### 2.2 Create Project Structure

```bash
# Create directories
mkdir -p src config tests data exports .github/workflows

# Create placeholder files
touch src/__init__.py
touch tests/__init__.py
touch data/.gitkeep
touch exports/.gitkeep
```

### 2.3 Setup Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### 2.4 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 📋 Step 3: Create Essential Files

### 3.1 Create `.gitignore`

```bash
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.venv/

# Database files
*.db
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Data directories
data/*
!data/.gitkeep
exports/*
!exports/.gitkeep

# Logs
*.log
logs/

# Environment variables
.env
.env.local

# Streamlit
.streamlit/secrets.toml

# Jupyter
.ipynb_checkpoints/

# Test coverage
.coverage
htmlcov/

# Distribution
dist/
build/
*.egg-info/

# Embeddings cache
embeddings_cache/
EOF
```

### 3.2 Create `requirements.txt`

```bash
cat > requirements.txt << 'EOF'
# Core Framework
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Web Crawling
aiohttp>=3.9.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Machine Learning & Embeddings
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
umap-learn>=0.5.4
hdbscan>=0.8.33

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
plotly>=5.18.0

# CLI
tqdm>=4.66.0
click>=8.1.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Development
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
EOF
```

### 3.3 Create `LICENSE`

```bash
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

---

## 📋 Step 4: Add Source Code Files

### 4.1 Copy the main application files to `src/`

```bash
# Copy your app.py, cli.py, api.py to src/ directory
# If you downloaded them from this project:
cp app.py src/
cp cli.py src/
cp api.py src/
```

### 4.2 Create `config/settings.py`

```bash
cat > config/settings.py << 'EOF'
"""
Application Settings
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "internal_links.db"))

# Crawler settings
DEFAULT_USER_AGENT = "googlebot"
DEFAULT_CONCURRENT_REQUESTS = 10
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_REQUEST_DELAY = 0.5

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Clustering settings
DEFAULT_CLUSTERING_METHOD = "hdbscan"
DEFAULT_MIN_CLUSTER_SIZE = 5

# Suggestion settings
DEFAULT_TOP_K_SUGGESTIONS = 5
MIN_SIMILARITY_THRESHOLD = 0.3
EOF
```

### 4.3 Create `config/user_agents.py`

```bash
cat > config/user_agents.py << 'EOF'
"""
User Agent Strings for Web Crawling
"""

USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_mobile': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'firefox': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
}

def get_user_agent(name: str) -> str:
    """Get user agent string by name"""
    return USER_AGENTS.get(name, USER_AGENTS['googlebot'])
EOF
```

---

## 📋 Step 5: Create GitHub Actions CI/CD

### 5.1 Create `.github/workflows/ci.yml`

```bash
mkdir -p .github/workflows

cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Run tests
      run: |
        pip install pytest pytest-asyncio
        pytest tests/ -v

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      run: |
        docker build -t internal-link-builder:latest .
    
    - name: Run Docker container test
      run: |
        docker run --rm internal-link-builder:latest python -c "import streamlit; print('Streamlit OK')"
EOF
```

---

## 📋 Step 6: Create Entry Point Scripts

### 6.1 Create `run_streamlit.py` (Root level)

```bash
cat > run_streamlit.py << 'EOF'
#!/usr/bin/env python3
"""
Entry point for Streamlit application
Run with: streamlit run run_streamlit.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the app
from src.app import main

if __name__ == "__main__":
    main()
EOF
```

### 6.2 Create `run_api.py` (Root level)

```bash
cat > run_api.py << 'EOF'
#!/usr/bin/env python3
"""
Entry point for FastAPI application
Run with: uvicorn run_api:app --reload
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

### 6.3 Update `Dockerfile`

```bash
cat > Dockerfile << 'EOF'
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data /app/exports

# Expose ports
EXPOSE 8501 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command (Streamlit)
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
```

### 6.4 Update `docker-compose.yml`

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Streamlit Web UI
  streamlit:
    build: .
    container_name: ilb-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./exports:/app/exports
    environment:
      - PYTHONPATH=/app
      - DATABASE_PATH=/app/data/internal_links.db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # FastAPI Backend (optional)
  api:
    build: .
    container_name: ilb-api
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./exports:/app/exports
    environment:
      - PYTHONPATH=/app
      - DATABASE_PATH=/app/data/internal_links.db
    restart: unless-stopped
    profiles:
      - api  # Only starts with: docker-compose --profile api up

volumes:
  data:
  exports:
EOF
```

---

## 📋 Step 7: Commit & Push to GitHub

```bash
# Add all files to git
git add .

# Commit
git commit -m "Initial commit: Internal Link Builder v1.0

Features:
- Async web crawler with Googlebot UA
- Semantic embeddings with sentence-transformers
- HDBSCAN/K-means clustering
- UMAP visualization
- AI-powered link suggestions
- Orphan page detection
- Cannibalization detection
- CLI, API, and Streamlit UI"

# Push to GitHub
git push origin main
```

---

## 📋 Step 8: Running the Application

### Option 1: Local Development (Streamlit)

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run Streamlit
streamlit run src/app.py

# Open browser: http://localhost:8501
```

### Option 2: Local Development (FastAPI)

```bash
# Run FastAPI
uvicorn src.api:app --reload --port 8000

# API docs: http://localhost:8000/docs
```

### Option 3: CLI Tool

```bash
# Crawl URLs
python src/cli.py crawl urls.txt --user-agent googlebot --concurrent 20

# Generate embeddings
python src/cli.py embed

# Cluster pages
python src/cli.py cluster --method hdbscan

# Generate suggestions
python src/cli.py suggest --output suggestions.csv

# View stats
python src/cli.py stats
```

### Option 4: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run with API service
docker-compose --profile api up --build

# Access:
# Streamlit: http://localhost:8501
# API: http://localhost:8000
```

---

## 📋 Step 9: Deployment Options

### Option A: Deploy to Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select:
   - Repository: `your-username/internal-link-builder`
   - Branch: `main`
   - Main file path: `src/app.py`
5. Click "Deploy"

### Option B: Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### Option C: Deploy to Render

1. Go to [render.com](https://render.com)
2. Create new "Web Service"
3. Connect GitHub repository
4. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`

### Option D: Deploy to AWS/GCP/Azure

See the Docker configuration for containerized deployment to any cloud platform.

---

## 📋 Step 10: Create Sample Test File

### `tests/test_crawler.py`

```bash
cat > tests/test_crawler.py << 'EOF'
"""
Tests for the web crawler
"""
import pytest
import asyncio
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCrawler:
    """Test cases for AsyncCrawler"""
    
    def test_user_agents_available(self):
        """Test that user agents are defined"""
        from config.user_agents import USER_AGENTS
        
        assert 'googlebot' in USER_AGENTS
        assert 'bingbot' in USER_AGENTS
        assert 'chrome' in USER_AGENTS
    
    def test_parse_simple_html(self):
        """Test HTML parsing"""
        html = """
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is the body content.</p>
            <a href="/page2">Link to page 2</a>
        </body>
        </html>
        """
        
        # Test that BeautifulSoup can parse
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        assert soup.find('title').text == 'Test Page'
        assert soup.find('h1').text == 'Main Heading'
        assert len(soup.find_all('a')) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
EOF
```

---

## 🎯 Quick Start Commands Summary

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/internal-link-builder.git
cd internal-link-builder

# 2. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run
streamlit run src/app.py

# 4. Use CLI
python src/cli.py crawl urls.txt --user-agent googlebot
python src/cli.py embed
python src/cli.py cluster
python src/cli.py suggest --output suggestions.csv

# 5. Docker
docker-compose up --build
```

---

## 🆘 Troubleshooting

### Issue: "Module not found"
```bash
# Make sure you're in virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Permission denied" on Linux/Mac
```bash
chmod +x src/cli.py
```

### Issue: Streamlit port already in use
```bash
streamlit run src/app.py --server.port 8502
```

### Issue: Docker build fails
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [HDBSCAN Clustering](https://hdbscan.readthedocs.io/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

---

**Happy Internal Linking! 🔗**
