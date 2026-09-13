#!/usr/bin/env python3
"""
Deploy the 3D Neural Architecture Visualizer as a static Hugging Face Space under the Hooshaai organization.
Space URL: https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer
"""

import os
import sys
import shutil
import tempfile
from huggingface_hub import HfApi

def deploy():
    org = "Hooshaai"
    space_name = "3d-neural-architecture-visualizer"
    repo_id = f"{org}/{space_name}"

    print(f"[*] Deploying 3D Neural Architecture Visualizer to Hugging Face Space: {repo_id} ...")
    api = HfApi()

    # 1. Create Space repository if it does not exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True
        )
        print(f"[OK] Space repository verified/created: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"[!] Note on repo creation: {e}")

    # 2. Prepare files in a clean staging directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_copy = [
        "index.html",
        "style.css",
        "app.js",
        "spatial-data.json",
        "modules-catalog.json",
        "modules_components.json",
        "favicon.ico",
        "standalone.html"
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[*] Preparing deployment bundle in {tmpdir} ...")

        # Copy assets
        for f in files_to_copy:
            src = os.path.join(base_dir, f)
            dst = os.path.join(tmpdir, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                sz_kb = os.path.getsize(dst) / 1024
                print(f"  + {f} ({sz_kb:.1f} KB)")
            else:
                print(f"  - WARNING: {f} not found!")

        # Create Space README.md with YAML metadata
        readme_content = """---
title: 3D Neural Architecture Visualizer
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
---

# 3D Neural Architecture Visualizer & Compression Atlas

Interactive WebGL 3D dataflow and multi-model architectural comparison across **DistilBERT**, **RoBERTa**, **GPT-2**, and **Qwen-2.5**.

### Key Capabilities
- **Interactive 3D Computational Graphs**: Explore exact projection geometries, attention heads, SVD bottlenecks, GPT-2 fused Conv1D, and Qwen-2.5 GQA / SwiGLU routing.
- **Physical Token Trajectories**: Fluid token streams physically matching PyTorch forward passes and residual skip highways.
- **Inter-Module Wiring Conduits**: Luminous 3D dataflow buses and fan-out / fan-in connections between consecutive layers.
- **Verified Benchmark Matrix**: Real GLUE verification results (accuracy %, peak VRAM MB, throughput tokens/sec) for all 4 models.
- **240+ Neural Modules Catalog**: Detailed component breakdown, mathematical formulations, and compression metrics.

---
*Maintained by [Hooshaai](https://huggingface.co/Hooshaai).*
"""
        with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"[*] Uploading deployment bundle to {repo_id} ...")
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy 3D Neural Architecture Visualizer with dataflow conduits and verified benchmarks"
        )

        print(f"\n[SUCCESS] Successfully deployed to Hugging Face Spaces!")
        print(f"👉 Direct URL: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy()
