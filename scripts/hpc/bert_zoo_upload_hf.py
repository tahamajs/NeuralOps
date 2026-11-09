#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dual-Mode Hugging Face Hub Uploader & Disk Quota Watcher.

Can be executed as:
  1. A standalone watcher on the HPC login node (where internet access is unrestricted):
     python3 scripts/hpc/bert_zoo_upload_hf.py --watch_dir results/bert_attention_zoo --cleanup 1
  2. A direct single-directory uploader:
     python3 scripts/hpc/bert_zoo_upload_hf.py --single_dir results/bert_attention_zoo/gated_delta_net
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bert_zoo_uploader")


def upload_directory(
    target_dir: Path,
    hf_org: str,
    cleanup_weights: bool = True,
    token: str | None = None,
) -> bool:
    """Uploads one checkpoint directory to Hugging Face Hub and deletes heavy weights."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN is unset. Cannot upload to Hugging Face Hub.")
        return False

    attn_key = target_dir.name
    slug = attn_key.replace("_", "-")

    # Read metrics if present
    metrics_file = target_dir / "training_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    is_compressor = "compressor" in str(target_dir) or "compressor" in metrics or "param_reduction" in metrics
    is_qwen = "qwen" in str(target_dir) or "qwen" in metrics or "model" in metrics and "qwen" in str(metrics["model"]).lower()

    if is_compressor:
        if is_qwen:
            repo_id = f"{hf_org}/qwen-comp-{slug}-wiki"
        else:
            repo_id = f"{hf_org}/bert-comp-{slug}-wiki"
    elif is_qwen:
        repo_id = f"{hf_org}/qwen-2.5-0.5b-{slug}-wiki"
    else:
        repo_id = f"{hf_org}/bert-{slug}-wiki"

    logger.info("[%s] Uploading to %s...", attn_key, repo_id)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        commit_info = api.upload_folder(
            folder_path=str(target_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Distill bert-{slug} on WikiText-2",
        )
        commit_hash = getattr(commit_info, "commit_id", "verified")
        logger.info("[%s] Upload successful! Remote commit: %s", attn_key, commit_hash)

        # Write confirmation record
        success_record = {
            "status": "UPLOAD_SUCCESS",
            "repo_id": repo_id,
            "commit_hash": str(commit_hash),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        (target_dir / "upload_success.json").write_text(json.dumps(success_record, indent=2))

        # Cleanup heavy weights
        if cleanup_weights:
            for w_name in ["model.safetensors", "pytorch_model.bin", "optimizer.pt"]:
                w_file = target_dir / w_name
                if w_file.exists():
                    size_mb = w_file.stat().st_size / (1024 * 1024)
                    os.remove(w_file)
                    logger.info("[%s] [DISK CLEANUP] Removed %s (%.1f MB freed).", attn_key, w_name, size_mb)

        return True

    except Exception as e:
        logger.error("[%s] Upload failed: %s", attn_key, e, exc_info=True)
        (target_dir / "upload_error.json").write_text(
            json.dumps({"error": str(e), "timestamp": datetime.datetime.utcnow().isoformat() + "Z"})
        )
        return False


def watch_and_upload(
    watch_dir: Path,
    hf_org: str,
    cleanup_weights: bool = True,
    poll_interval: int = 30,
) -> None:
    """Continuously poll watch_dir for completed models and upload them."""
    logger.info("Watching %s for completed attention models (poll=%ds)...", watch_dir, poll_interval)

    while True:
        if not watch_dir.exists():
            time.sleep(poll_interval)
            continue

        for sub in sorted(watch_dir.iterdir()):
            if not sub.is_dir() or sub.name.startswith((".", "upload_")):
                continue

            # A model is complete if training_metrics.json exists
            is_complete = (sub / "training_metrics.json").exists()
            already_uploaded = (sub / "upload_success.json").exists()

            if is_complete and not already_uploaded:
                upload_directory(sub, hf_org, cleanup_weights=cleanup_weights)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT Attention Zoo Hugging Face Uploader")
    parser.add_argument("--watch_dir", default=None, help="Directory to watch for completed models")
    parser.add_argument("--single_dir", default=None, help="Single directory to upload directly")
    parser.add_argument("--hf_org", default="Hooshaai", help="Hugging Face organization")
    parser.add_argument("--cleanup", type=int, default=1, help="1 to delete weights after upload, 0 to keep")
    parser.add_argument("--poll_interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    if args.single_dir:
        path = Path(args.single_dir)
        if not path.is_dir():
            logger.error("Specified path is not a directory: %s", path)
            sys.exit(1)
        success = upload_directory(path, args.hf_org, cleanup_weights=bool(args.cleanup))
        sys.exit(0 if success else 1)

    if args.watch_dir:
        watch_and_upload(Path(args.watch_dir), args.hf_org, cleanup_weights=bool(args.cleanup), poll_interval=args.poll_interval)
    else:
        logger.error("Must specify either --watch_dir or --single_dir")
        sys.exit(2)


if __name__ == "__main__":
    main()
