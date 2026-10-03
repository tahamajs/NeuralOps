# 3D Neural Architecture Visualizer · Deployment Guide

Zero-dependency deployment options for the 3D Neural Architecture Atlas.

---

## Option 1: One-Click Instant Local Launch (Easiest)

We built an automatic launcher script [`serve.py`](serve.py) that starts a clean local server and opens your browser:

```bash
python3 Docs/visualizer/serve.py
```

---

## Option 2: 100% Offline Standalone File (No Server Needed!)

We compiled a standalone single-file distribution: [`standalone.html`](standalone.html).
- All CSS, model spatial data, and Three.js visualizer code are bundled inside.
- **You can double-click `standalone.html` directly in your file manager** or run:
  ```bash
  open Docs/visualizer/standalone.html
  ```

---

## Option 3: GitHub Pages (Free Cloud Hosting)

1. In your GitHub repository settings, navigate to **Pages**.
2. Set the Source branch to `main` and folder to `/Docs` (or push `Docs/visualizer` to a `gh-pages` branch).
3. Your visualizer will be live at:
   `https://<username>.github.io/<repo>/visualizer/index.html`

---

## Option 4: Vercel / Netlify / Cloudflare Pages

Since the visualizer is pure static HTML/JS/CSS with no build step required:
- **Framework Preset**: `Other` (Static)
- **Root Directory**: `Docs/visualizer`
- **Build Command**: *(None / Leave Empty)*
- **Output Directory**: `.`
