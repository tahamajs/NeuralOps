#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI utility to synchronize NeuralOps Hugging Face Space documentation and static WebGL assets.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize Hugging Face Space and profile READMEs for Hooshaai."
    )
    parser.add_argument(
        "--space-repo",
        type=str,
        default="Hooshaai/README",
        help="Target Hugging Face Space repository ID (default: Hooshaai/README).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token with write permissions (or set via HF_TOKEN env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without performing actual uploads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    space_bundle_dir = repo_root / "docs" / "huggingface" / "space_bundle"

    readme_path = space_bundle_dir / "README.md"
    index_path = space_bundle_dir / "index.html"
    manifold_path = space_bundle_dir / "latent_manifold.html"

    if not readme_path.exists():
        print(f"Error: {readme_path} does not exist.", file=sys.stderr)
        return 1

    if not index_path.exists():
        print(f"Error: {index_path} does not exist.", file=sys.stderr)
        return 1

    print(f"Preparing to sync space: {args.space_repo}")
    print(f"  README:   {readme_path} ({readme_path.stat().st_size} bytes)")
    print(f"  index:    {index_path} ({index_path.stat().st_size} bytes)")
    if manifold_path.exists():
        print(f"  manifold: {manifold_path} ({manifold_path.stat().st_size} bytes)")

    if args.dry_run:
        print("[DRY-RUN] Files validated. Skipping upload.")
        return 0

    if not args.token:
        print(
            "Note: No Hugging Face token provided. You can provide one via --token or HF_TOKEN env var.",
            file=sys.stderr,
        )
        print("Alternative manual Git deployment command:", file=sys.stderr)
        print(f"  git clone https://huggingface.co/spaces/{args.space_repo} /tmp/hf_space", file=sys.stderr)
        print(f"  cp {space_bundle_dir}/* /tmp/hf_space/", file=sys.stderr)
        print("  cd /tmp/hf_space && git add . && git commit -m 'feat: update space' && git push", file=sys.stderr)
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)

    print(f"Uploading files to Hugging Face Space '{args.space_repo}'...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.space_repo,
        repo_type="space",
        commit_message="docs: sync NeuralOps Space README and metadata",
    )
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo="index.html",
        repo_id=args.space_repo,
        repo_type="space",
        commit_message="feat: deploy 3D Neural Architecture Atlas standalone WebGL",
    )
    if manifold_path.exists():
        api.upload_file(
            path_or_fileobj=str(manifold_path),
            path_in_repo="latent_manifold.html",
            repo_id=args.space_repo,
            repo_type="space",
            commit_message="feat: upload latent manifold visualizer page",
        )

    print(f"Successfully uploaded space bundle to https://huggingface.co/spaces/{args.space_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
