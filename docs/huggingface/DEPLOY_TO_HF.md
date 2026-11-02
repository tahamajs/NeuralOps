# 🚀 Hugging Face Deployment & Synchronization Guide

Step-by-step instructions for deploying and synchronizing the **Hooshaai** organization profile, model cards, benchmark datasets, and interactive 3D WebGL Spaces on Hugging Face Hub.

---

## 📋 Available Documentation Files

| File | Purpose | Target Destination on Hugging Face |
| :--- | :--- | :--- |
| [`HOOSHAAI_PROFILE_README.md`](HOOSHAAI_PROFILE_README.md) | Official Organization Overview & Profile | `https://huggingface.co/Hooshaai` (Organization Profile / Overview) |
| [`SPACE_README.md`](SPACE_README.md) | Hugging Face Space Metadata & Documentation | `https://huggingface.co/spaces/Hooshaai/README` (General Space) |
| [`spaces/3d-neural-architecture-visualizer/`](spaces/3d-neural-architecture-visualizer/) | Complete 3D Visualizer WebGL Space Distribution | `https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer` |

---

## 🛠️ Method 1: Instant Web UI Update (Fastest)

### For the Space (`https://huggingface.co/spaces/Hooshaai/README`):
1. Navigate to: **[https://huggingface.co/spaces/Hooshaai/README](https://huggingface.co/spaces/Hooshaai/README)**
2. Click on the **Files and versions** tab.
3. Click on `README.md` (or click **Add file** -> **Create a new file** if it doesn't exist).
4. Copy the entire contents of [`docs/huggingface/SPACE_README.md`](SPACE_README.md) and paste it into the editor.
5. In addition, to enable the interactive 3D WebGL Atlas in the Space:
   - Upload `docs/visualizer/standalone.html` renamed to `index.html` (or upload `index.html`, `style.css`, `app.js`, `spatial-data.json`, `modules-catalog.json` from `docs/visualizer/`).
6. Click **Commit changes to main**.
7. The Space will automatically build and serve the interactive 3D Atlas and documentation within seconds!

### For the Organization Profile (`https://huggingface.co/Hooshaai`):
1. Log in to Hugging Face with organization admin credentials.
2. Navigate to: **[https://huggingface.co/organizations/Hooshaai/settings](https://huggingface.co/organizations/Hooshaai/settings)** (or your organization page).
3. If using an organization card / repository, edit `README.md` with the contents of [`docs/huggingface/HOOSHAAI_PROFILE_README.md`](HOOSHAAI_PROFILE_README.md).
4. Save and publish.

---

## 💻 Method 2: Command-Line Deployment via Git

### 1. Clone the Hugging Face Space
```bash
# Clone the Space repository via HTTPS or SSH
git clone https://huggingface.co/spaces/Hooshaai/README /tmp/hf-hooshaai-space
cd /tmp/hf-hooshaai-space
```

### 2. Copy the Documentation & 3D WebGL Assets
```bash
# Copy the Space README
cp /Users/tahamajs/Documents/uni/Project/docs/huggingface/SPACE_README.md ./README.md

# Copy the standalone 3D Visualizer as the root index.html
cp /Users/tahamajs/Documents/uni/Project/docs/visualizer/standalone.html ./index.html

# Copy supplementary assets (optional for multi-page visualizer)
cp /Users/tahamajs/Documents/uni/Project/docs/visualizer/latent_manifold.html ./latent_manifold.html
```

### 3. Commit and Push
```bash
git add README.md index.html latent_manifold.html
git commit --author="taha <tahamaj4@gmail.com>" -m "feat: deploy NeuralOps v1.7.0 3D Atlas Space and documentation"
git push origin main
```

---

## 🐍 Method 3: Automated Upload via `huggingface_hub` Python API

```python
from huggingface_hub import HfApi

api = HfApi()

# 1. Upload the Space README
api.upload_file(
    path_or_fileobj="docs/huggingface/SPACE_README.md",
    path_in_repo="README.md",
    repo_id="Hooshaai/README",
    repo_type="space",
    commit_message="docs: update NeuralOps Space README and metadata",
)

# 2. Upload the Standalone 3D Atlas as index.html
api.upload_file(
    path_or_fileobj="docs/visualizer/standalone.html",
    path_in_repo="index.html",
    repo_id="Hooshaai/README",
    repo_type="space",
    commit_message="feat: upload interactive 3D WebGL Neural Architecture Atlas",
)

print("✅ Successfully deployed to https://huggingface.co/spaces/Hooshaai/README")
```

---

## 🌌 Deploying `Hooshaai/3d-neural-architecture-visualizer`

The dedicated 3D visualizer space package is located at `docs/huggingface/spaces/3d-neural-architecture-visualizer/`.

### 1-Click Automated Script:
```bash
export HF_TOKEN="hf_your_write_token"
python3 docs/huggingface/spaces/3d-neural-architecture-visualizer/deploy_to_hf_space.py
```

### Or using Bash Git Sync:
```bash
export HF_TOKEN="hf_your_write_token"
./docs/huggingface/spaces/3d-neural-architecture-visualizer/sync_space.sh
```

---

## 🔍 Verification Checklist

After deploying, verify the following URLs:
- [x] **General Space URL**: [https://huggingface.co/spaces/Hooshaai/README](https://huggingface.co/spaces/Hooshaai/README) (Check that the Space shows the ⚡ emoji and builds successfully).
- [x] **3D Visualizer Space URL**: [https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer](https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer) (Interactive WebGL 3D atlas with 🌌 emoji).
- [x] **Direct Space App**: [https://hooshaai-3d-neural-architecture-visualizer.hf.space](https://hooshaai-3d-neural-architecture-visualizer.hf.space).
- [x] **Organization Page**: [https://huggingface.co/Hooshaai](https://huggingface.co/Hooshaai) (Verify 14 model checkpoints and dataset).
- [x] **Dataset Ledger**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results) (Verify parquet, CSV, and SQLite files).
