#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated deployment script to synchronize the 3D Neural Architecture Visualizer
static WebGL application and metadata to Hugging Face Spaces:
https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer
"""

import argparse
import os
import sys
from pathlib import Path


def deploy(
    repo_id: str = "Hooshaai/3d-neural-architecture-visualizer",
    token: str | None = None,
    dry_run: bool = False,
) -> None:
    """Uploads the local 3D visualizer static bundle to Hugging Face Spaces.

    Args:
        repo_id: The target Hugging Face Space repository identifier.
        token: Hugging Face API user access token (reads HF_TOKEN env var if None).
        dry_run: If True, lists files to be uploaded without uploading.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "Error: 'huggingface_hub' is not installed. Please install it with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    source_dir = Path(__file__).resolve().parent
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    print("=" * 72)
    print("🚀 NeuralOps Hugging Face Space Deployment")
    print(f"   Target Space: https://huggingface.co/spaces/{repo_id}")
    print(f"   Source Directory: {source_dir}")
    print(f"   Dry Run: {dry_run}")
    print("=" * 72)

    # Gather all files in this space package
    all_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith((".pyc", ".DS_Store")):
                continue
            rel_path = Path(root, f).relative_to(source_dir)
            all_files.append(rel_path)

    all_files = sorted(all_files)
    print(f"\n📦 Discovered {len(all_files)} files in static distribution bundle:")
    for path in all_files[:15]:
        size_kb = (source_dir / path).stat().st_size / 1024.0
        print(f"   - {path} ({size_kb:.1f} KB)")
    if len(all_files) > 15:
        print(f"   ... and {len(all_files) - 15} more files.")

    if dry_run:
        print("\n[DRY RUN] No files were uploaded.")
        return

    api = HfApi(token=hf_token)

    # Ensure repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"\n✅ Target space '{repo_id}' found.")
    except Exception as e:
        print(f"\nℹ️ Space '{repo_id}' not found or inaccessible: {e}")
        print(f"Attempting to create space '{repo_id}' with static SDK...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="static",
                private=False,
            )
            print(f"✅ Space '{repo_id}' created successfully.")
        except Exception as create_err:
            print(f"⚠️ Could not create space automatically: {create_err}")
            print("Please ensure your token has WRITE permissions to Hooshaai organization.")

    print(f"\n📤 Uploading folder '{source_dir}' to Space '{repo_id}'...")
    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="feat: deploy 3D Neural Architecture Visualizer (241 modules)",
        ignore_patterns=["*.pyc", ".DS_Store", "__pycache__/*"],
    )

    print("\n" + "=" * 72)
    print("✨ Deployment successfully completed!")
    print(f"   Live Space URL: https://huggingface.co/spaces/{repo_id}")
    print("   Direct Static URL: https://hooshaai-3d-neural-architecture-visualizer.hf.space")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy 3D Neural Architecture Visualizer to Hugging Face Spaces."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Hooshaai/3d-neural-architecture-visualizer",
        help="Target Hugging Face Space repository (default: Hooshaai/3d-neural-architecture-visualizer)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API User Access Token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the deployment and list files without uploading",
    )
    args = parser.parse_args()

    deploy(
        repo_id=args.repo_id,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
