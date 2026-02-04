# 🚀 Deploy to Streamlit Cloud - Complete Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (free)
- Your code pushed to GitHub

---

## Step 1: Push Code to GitHub

### 1.1 Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `internal-link-builder`
3. Set to **Public** (required for free Streamlit hosting)
4. Click **Create repository**

### 1.2 Push Your Code

```bash
# Navigate to your project folder
cd internal-link-builder

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Internal Link Builder"

# Add remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/internal-link-builder.git

# Push to GitHub
git push -u origin main
```

---

## Step 2: Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up"** or **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub

---

## Step 3: Deploy Your App

### 3.1 Create New App

1. Click **"New app"** button
2. Fill in the form:

| Field | Value |
|-------|-------|
| **Repository** | `YOUR_USERNAME/internal-link-builder` |
| **Branch** | `main` |
| **Main file path** | `src/app.py` |

3. Click **"Advanced settings"** (optional)

### 3.2 Advanced Settings (Optional)

**Python version:** `3.11`

**Secrets** (if needed):
```toml
# Add any API keys here
# ANTHROPIC_API_KEY = "your-key"
```

### 3.3 Deploy

1. Click **"Deploy!"**
2. Wait 5-10 minutes for first deployment
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## Step 4: Required Files for Streamlit Cloud

Make sure these files exist in your repository:

### 4.1 `requirements.txt` (already created)

```
streamlit>=1.28.0
aiohttp>=3.9.0
beautifulsoup4>=4.12.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
umap-learn>=0.5.4
hdbscan>=0.8.33
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
lxml>=4.9.0
```

### 4.2 `.streamlit/config.toml` (Create this)

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false
```

### 4.3 `packages.txt` (Create if needed for system packages)

```
# System packages (if needed)
# libgl1-mesa-glx
```

---

## Step 5: Create Streamlit Config File

Run this to create the config:

```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true

[browser]
gatherUsageStats = false
EOF
```

---

## Step 6: Troubleshooting

### Issue: "Module not found"
**Solution:** Make sure all dependencies are in `requirements.txt`

### Issue: App crashes on startup
**Solution:** Check the logs in Streamlit Cloud dashboard

### Issue: File path errors
**Solution:** Use relative paths or `Path(__file__).parent`

### Issue: Memory limit exceeded
**Solution:** 
- Streamlit Cloud free tier has 1GB RAM limit
- Reduce batch sizes in embeddings
- Use smaller embedding models

### Issue: App takes too long to load
**Solution:**
- Add caching with `@st.cache_data` or `@st.cache_resource`
- Pre-compute embeddings

---

## Step 7: Optimize for Streamlit Cloud

### 7.1 Add Caching

Add this to your `src/app.py`:

```python
import streamlit as st

@st.cache_resource
def load_embedding_model():
    """Cache the embedding model"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def generate_embeddings_cached(texts):
    """Cache embeddings generation"""
    model = load_embedding_model()
    return model.encode(texts)
```

### 7.2 Add Loading Spinner

```python
with st.spinner('Loading model... This may take a minute on first run.'):
    model = load_embedding_model()
```

---

## Step 8: Custom Domain (Optional)

1. Go to your app settings in Streamlit Cloud
2. Click **"Custom domain"**
3. Add your domain (e.g., `links.yourdomain.com`)
4. Add CNAME record in your DNS:
   - Type: `CNAME`
   - Name: `links`
   - Value: `YOUR_APP.streamlit.app`

---

## Step 9: Monitor Your App

### View Logs
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click **"Manage app"** → **"Logs"**

### View Analytics
1. Click **"Analytics"** in app settings
2. See visitor counts and usage

---

## Quick Deploy Checklist

- [ ] Code pushed to GitHub (public repo)
- [ ] `requirements.txt` in root directory
- [ ] Main file path is `src/app.py`
- [ ] No hardcoded file paths (use relative paths)
- [ ] Database uses SQLite (file-based)
- [ ] Added caching for heavy operations
- [ ] Tested locally with `streamlit run src/app.py`

---

## Example: Complete Deployment Commands

```bash
# 1. Clone or navigate to your project
cd internal-link-builder

# 2. Create Streamlit config
mkdir -p .streamlit
echo '[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false' > .streamlit/config.toml

# 3. Test locally
streamlit run src/app.py

# 4. Add to git
git add .
git commit -m "Add Streamlit config"
git push origin main

# 5. Go to share.streamlit.io and deploy!
```

---

## Your App URL

After deployment, your app will be available at:

```
https://YOUR_APP_NAME.streamlit.app
```

Or with custom subdomain:
```
https://internal-link-builder-YOUR_USERNAME.streamlit.app
```

---

## Free Tier Limits

| Resource | Limit |
|----------|-------|
| Apps | Unlimited public apps |
| RAM | 1 GB |
| CPU | Shared |
| Storage | Ephemeral (resets on reboot) |
| Bandwidth | Unlimited |

---

## Need More Resources?

Consider upgrading to:
- **Streamlit Teams** - More resources, private apps
- **Self-hosted** - Use Docker on your own server
- **Cloud VPS** - Deploy on DigitalOcean, AWS, etc.

---

**Happy Deploying! 🎉**
