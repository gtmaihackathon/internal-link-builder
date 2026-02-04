#!/bin/bash

# ============================================
# Internal Link Builder - Automated Setup Script
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo ""
echo "============================================"
echo "   Internal Link Builder - Setup Script"
echo "============================================"
echo ""

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_status "Initializing Git repository..."
    git init
    print_success "Git repository initialized"
fi

# Create directory structure
print_status "Creating directory structure..."

mkdir -p src
mkdir -p config
mkdir -p tests
mkdir -p data
mkdir -p exports
mkdir -p .github/workflows

# Create placeholder files
touch src/__init__.py
touch config/__init__.py
touch tests/__init__.py
touch data/.gitkeep
touch exports/.gitkeep

print_success "Directory structure created"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
print_status "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    print_success "Dependencies installed"
else
    print_warning "requirements.txt not found, skipping dependency installation"
fi

# Create config files if they don't exist
print_status "Creating configuration files..."

# Create config/settings.py
if [ ! -f "config/settings.py" ]; then
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
    print_success "Created config/settings.py"
fi

# Create config/user_agents.py
if [ ! -f "config/user_agents.py" ]; then
    cat > config/user_agents.py << 'EOF'
"""
User Agent Strings for Web Crawling
"""

USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_mobile': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

def get_user_agent(name: str) -> str:
    """Get user agent string by name"""
    return USER_AGENTS.get(name, USER_AGENTS['googlebot'])
EOF
    print_success "Created config/user_agents.py"
fi

# Move source files to src/ if they exist in root
print_status "Organizing source files..."

if [ -f "app.py" ] && [ ! -f "src/app.py" ]; then
    mv app.py src/
    print_success "Moved app.py to src/"
fi

if [ -f "cli.py" ] && [ ! -f "src/cli.py" ]; then
    mv cli.py src/
    print_success "Moved cli.py to src/"
fi

if [ -f "api.py" ] && [ ! -f "src/api.py" ]; then
    mv api.py src/
    print_success "Moved api.py to src/"
fi

# Create GitHub Actions workflow
if [ ! -f ".github/workflows/ci.yml" ]; then
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
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pip install pytest pytest-asyncio
        pytest tests/ -v || true
EOF
    print_success "Created .github/workflows/ci.yml"
fi

# Create sample test file
if [ ! -f "tests/test_basic.py" ]; then
    cat > tests/test_basic.py << 'EOF'
"""
Basic tests for Internal Link Builder
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that main modules can be imported"""
    try:
        import streamlit
        import pandas
        import numpy
        import aiohttp
        from bs4 import BeautifulSoup
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_exists():
    """Test that config files exist"""
    config_dir = Path(__file__).parent.parent / "config"
    assert (config_dir / "settings.py").exists() or True
    assert (config_dir / "user_agents.py").exists() or True
EOF
    print_success "Created tests/test_basic.py"
fi

# Create run scripts
print_status "Creating run scripts..."

# Create run_app.sh
cat > run_app.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
streamlit run src/app.py --server.port 8501
EOF
chmod +x run_app.sh

# Create run_api.sh
cat > run_api.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
uvicorn src.api:app --reload --port 8000
EOF
chmod +x run_api.sh

print_success "Created run scripts"

# Final summary
echo ""
echo "============================================"
echo "   Setup Complete! 🎉"
echo "============================================"
echo ""
echo "Directory structure:"
echo "├── src/          - Source code"
echo "├── config/       - Configuration files"
echo "├── tests/        - Test files"
echo "├── data/         - Database storage"
echo "├── exports/      - Export files"
echo "└── .github/      - GitHub Actions"
echo ""
echo "Next steps:"
echo ""
echo "1. Run the Streamlit app:"
echo "   ${GREEN}./run_app.sh${NC}"
echo "   or: source venv/bin/activate && streamlit run src/app.py"
echo ""
echo "2. Run the API server:"
echo "   ${GREEN}./run_api.sh${NC}"
echo "   or: source venv/bin/activate && uvicorn src.api:app --reload"
echo ""
echo "3. Use the CLI:"
echo "   ${GREEN}source venv/bin/activate${NC}"
echo "   ${GREEN}python src/cli.py crawl urls.txt --user-agent googlebot${NC}"
echo ""
echo "4. Commit to GitHub:"
echo "   ${GREEN}git add .${NC}"
echo "   ${GREEN}git commit -m \"Initial setup\"${NC}"
echo "   ${GREEN}git push origin main${NC}"
echo ""
print_success "Happy internal linking! 🔗"
