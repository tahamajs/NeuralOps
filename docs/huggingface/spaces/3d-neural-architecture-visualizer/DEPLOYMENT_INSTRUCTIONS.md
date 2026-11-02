# 🌌 3D Neural Architecture Visualizer · Deployment Guide
### Hugging Face Space: `Hooshaai/3d-neural-architecture-visualizer`
URL: **https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer**

---

## 📋 Overview

This folder contains the complete, self-contained static web application distribution ready for deployment to Hugging Face Spaces (`sdk: static`).

### Included Files & Bundles:
- `README.md`: Hugging Face Space metadata header (`sdk: static`, `emoji: 🌌`, color gradient) and comprehensive scientific presentation.
- `index.html`: Main interactive WebGL 3D computational graph visualizer.
- `app.js`: Modular bootstrap script.
- `style.css`: Visualizer styling, controls, dark mode HUD.
- `spatial-data.json`: 3D node coordinates, tensors, and connection highways.
- `modules-catalog.json`: 241 research module catalog with LaTeX formulas and tensor shapes.
- `modules_components.json`: Hierarchical component definitions.
- `standalone.html`: 100% offline self-contained single-file visualizer.
- `latent_manifold.html`: 3D Latent Manifold & SVD Spectral Visualizer.
- `favicon.ico`: Application icon.
- `src/`: Modular ES6 scripts (3D renderers, physics engine, UI panels).
- `components/`: 200+ detailed component JSON descriptors.
- `deploy_to_hf_space.py`: Python CLI deployment tool using `huggingface_hub`.
- `sync_space.sh`: Bash script to clone, sync, and push via Git.

---

## 🛠️ Method 1: Instant Python Deployment (Recommended)

Run `deploy_to_hf_space.py`:

```bash
# Set your Hugging Face write token
export HF_TOKEN="hf_your_write_token_here"

# Execute deployment
python3 docs/huggingface/spaces/3d-neural-architecture-visualizer/deploy_to_hf_space.py
```

### Dry Run (Test without uploading):
```bash
python3 docs/huggingface/spaces/3d-neural-architecture-visualizer/deploy_to_hf_space.py --dry-run
```

---

## 💻 Method 2: Git Synchronization Script

Using `sync_space.sh`:

```bash
export HF_TOKEN="hf_your_write_token_here"
./docs/huggingface/spaces/3d-neural-architecture-visualizer/sync_space.sh
```

Or manually with git:

```bash
# 1. Clone the Space
git clone https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer /tmp/hf-visualizer
cd /tmp/hf-visualizer

# 2. Copy the distribution files
cp -R /Users/tahamajs/Documents/uni/Project/docs/huggingface/spaces/3d-neural-architecture-visualizer/* .

# 3. Commit and push
git config user.name "taha"
git config user.email "tahamaj4@gmail.com"
git add -A
git commit --author="taha <tahamaj4@gmail.com>" -m "feat: deploy 3D Neural Architecture Visualizer"
git push origin main
```

---

## 🌐 Method 3: Web UI Upload (No CLI needed)

1. Navigate to: **[https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer](https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer)**
2. Go to **Files and versions** -> **Add file** -> **Upload files**.
3. Drag and drop all files from `docs/huggingface/spaces/3d-neural-architecture-visualizer/` into the upload zone.
4. Commit changes directly to `main`.
5. Hugging Face will immediately build and serve the static application.

---

## ✅ Verification Checklist

Once deployed, visit:
- **Hugging Face Space**: [https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer](https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer)
- **Direct Fullscreen WebGL View**: [https://hooshaai-3d-neural-architecture-visualizer.hf.space](https://hooshaai-3d-neural-architecture-visualizer.hf.space)
- **Latent Manifold View**: [https://hooshaai-3d-neural-architecture-visualizer.hf.space/latent_manifold.html](https://hooshaai-3d-neural-architecture-visualizer.hf.space/latent_manifold.html)
- **Standalone Offline View**: [https://hooshaai-3d-neural-architecture-visualizer.hf.space/standalone.html](https://hooshaai-3d-neural-architecture-visualizer.hf.space/standalone.html)
