#!/usr/bin/env python3
"""
Upload and synchronize Docs and 3D Visualizer to Hugging Face Spaces & Hub.
- Space 1: Hooshaai/3d-neural-architecture-visualizer (Interactive 3D Visualizer)
- Space 2: Hooshaai/neuralops-docs (Complete Research Documentation Hub)
- Dataset: Hooshaai/svd-benchmark-master-results (Master benchmarks + metadata)
"""

import os
import sys
import json
import time
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN") or "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"
HF_ORG = "Hooshaai"
api = HfApi(token=TOKEN)

PROJECT_ROOT = Path("/Users/tahamajs/Documents/uni/Project")
DOCS_DIR = PROJECT_ROOT / "Docs"
VIS_DIR = DOCS_DIR / "visualizer"

def upload_visualizer_space():
    space_id = f"{HF_ORG}/3d-neural-architecture-visualizer"
    print(f"\n[1/3] Uploading 3D Neural Architecture Visualizer -> {space_id}...")
    
    # Files to ignore (test files, python caches, etc.)
    ignore_patterns = [
        "**/__pycache__/**",
        "*.pyc",
        ".DS_Store",
        "tests/**",
    ]
    
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            private=False
        )
        print(f"  Space {space_id} ready.")
    except Exception as e:
        print(f"  Repo status/create: {e}")

    try:
        res = api.upload_folder(
            folder_path=str(VIS_DIR),
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="feat(visualizer): upload complete enriched 240-module 3D atlas and ES engine",
        )
        print(f"  SUCCESS! 3D Visualizer Space synced -> https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"  ERROR uploading visualizer space: {e}")

def upload_docs_space():
    space_id = f"{HF_ORG}/neuralops-docs"
    print(f"\n[2/3] Uploading NeuralOps Research Docs Hub -> {space_id}...")
    
    # Ensure temporary README with YAML frontmatter if needed
    readme_path = DOCS_DIR / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    had_yaml = content.startswith("---")
    
    if not had_yaml:
        yaml_frontmatter = (
            "---\n"
            "title: NeuralOps & SVD Linear Attention Research Hub\n"
            "emoji: 📚\n"
            "colorFrom: indigo\n"
            "colorTo: purple\n"
            "sdk: static\n"
            "pinned: false\n"
            "license: mit\n"
            "---\n\n"
        )
        readme_path.write_text(yaml_frontmatter + content, encoding="utf-8")

    ignore_patterns = [
        "**/__pycache__/**",
        "*.pyc",
        ".DS_Store",
        "presentation/*.pptx",  # skip huge pptx in space
        "presentation/*.xlsx",
        "reports/historical/**",
    ]

    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            private=False
        )
        print(f"  Space {space_id} ready.")
    except Exception as e:
        print(f"  Repo status/create: {e}")

    try:
        res = api.upload_folder(
            folder_path=str(DOCS_DIR),
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="feat(docs): sync 100% verified documentation, catalog, and research guides",
        )
        print(f"  SUCCESS! Docs Space synced -> https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"  ERROR uploading docs space: {e}")

    if not had_yaml:
        # Restore clean local README without frontmatter
        readme_path.write_text(content, encoding="utf-8")

def upload_master_results():
    repo_id = f"{HF_ORG}/svd-benchmark-master-results"
    print(f"\n[3/3] Updating master results dataset -> {repo_id}...")
    
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=True
        )
    except Exception as e:
        pass

    files_to_upload = [
        ("Docs/catalog.json", "catalog.json"),
        ("Docs/DOCS_AUDIT_REPORT.md", "DOCS_AUDIT_REPORT.md"),
        ("Docs/visualizer/modules-catalog.json", "modules-catalog.json"),
        ("Docs/visualizer/modules_components.json", "modules_components.json"),
        ("Docs/results/master_benchmark_summary.json", "master_benchmark_summary.json"),
        ("Docs/results/master_benchmark_summary.csv", "master_benchmark_summary.csv"),
        ("Docs/results/all_histories.json", "all_histories.json"),
        ("Docs/results/pareto_frontier.json", "pareto_frontier.json"),
    ]

    for local_rel, repo_path in files_to_upload:
        local_p = PROJECT_ROOT / local_rel
        if local_p.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(local_p),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Update {repo_path} from verified audit",
                )
                print(f"  Uploaded {repo_path} to {repo_id}")
            except Exception as e:
                print(f"  Failed to upload {repo_path}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    upload_visualizer_space()
    upload_docs_space()
    upload_master_results()
    print(f"\nAll uploads completed in {time.time() - start_time:.1f}s.")
