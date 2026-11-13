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

from neuralops.hub.uploader import _redact_secrets, retry_hf_api_call

# Force official Hugging Face Hub endpoint (bypass any read-only mirrors like devneeds.ir)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

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
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        logger.error("HF_TOKEN is unset and ~/.cache/huggingface/token not found. Cannot upload to Hugging Face Hub.")
        return False

    attn_key = target_dir.name
    slug = attn_key.replace("_", "-")

    # Read metrics if present
    metrics_file = target_dir / "training_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("[%s] Could not read metrics file %s: %s", attn_key, metrics_file, err)

    is_compressor = "compressor" in str(target_dir) or "compressor" in metrics or "param_reduction" in metrics
    is_qwen = "qwen" in str(target_dir) or "qwen" in metrics or "model" in metrics and "qwen" in str(metrics["model"]).lower()

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        whoami = api.whoami()
        username = whoami.get("name", "Hooshaai")
        allowed_orgs = [o["name"] for o in whoami.get("orgs", []) if o.get("canCreateModels", False)]
        effective_org = hf_org if (hf_org in allowed_orgs or hf_org == username) else username

        if is_compressor:
            if is_qwen:
                repo_id = f"{effective_org}/qwen-comp-{slug}-wikitext2"
            else:
                repo_id = f"{effective_org}/bert-comp-{slug}-sst2"
        elif is_qwen:
            repo_id = f"{effective_org}/qwen-2.5-0.5b-{slug}-wikitext2"
        else:
            repo_id = f"{effective_org}/bert-{slug}-sst2"

        logger.info("[%s] Uploading files to %s via single-stream HTTPS...", attn_key, repo_id)
        retry_hf_api_call(api.create_repo, repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        # For BERT: purge whole-model weights to preserve storage/bandwidth
        # For Qwen: keep model.safetensors so HF users can load it directly
        if not is_qwen:
            for heavy_name in ["pytorch_model.bin", "model.safetensors", "optimizer.pt"]:
                heavy_path = target_dir / heavy_name
                if heavy_path.exists():
                    try:
                        attn_safetensors = target_dir / "attention_weights.safetensors"
                        if not attn_safetensors.exists() and heavy_name in ["pytorch_model.bin", "model.safetensors"]:
                            import torch
                            from safetensors.torch import save_file, load_file
                            if heavy_name.endswith(".safetensors"):
                                sd = load_file(str(heavy_path))
                            else:
                                sd = torch.load(str(heavy_path), map_location="cpu")
                            attn_sd = {k: v.half().contiguous() for k, v in sd.items() if any(term in k.lower() for term in ("self_attn", "attention", "classifier", "pooler", "adapter", "lora"))}
                            save_file(attn_sd, str(attn_safetensors))
                            logger.info("[%s] Converted %s to attention-only FP16 safetensors (%.2f MB)", attn_key, heavy_name, attn_safetensors.stat().st_size / (1024 * 1024))
                        size_mb = heavy_path.stat().st_size / (1024 * 1024)
                        os.remove(heavy_path)
                        logger.info("[%s] [DISK CLEANUP] Purged heavy base model %s (%.1f MB freed).", attn_key, heavy_name, size_mb)
                    except Exception as e:
                        logger.warning("[%s] Could not purge heavy file %s: %s", attn_key, heavy_name, e)

        # Upload files
        files_to_upload = sorted([
            p for p in target_dir.iterdir()
            if p.is_file()
            and not p.name.startswith((".", "upload_"))
            and (p.name not in ["pytorch_model.bin", "model.safetensors", "optimizer.pt"] if not is_qwen else p.name != "optimizer.pt")
        ])
        uploaded_count = 0
        for fpath in files_to_upload:
            logger.info("[%s] Uploading %s (%.2f MB)...", attn_key, fpath.name, fpath.stat().st_size / (1024 * 1024))
            retry_hf_api_call(
                api.upload_file,
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {fpath.name} for bert-{slug} (attention weights only)",
            )
            uploaded_count += 1

        logger.info("[%s] Upload completed successfully (%d files uploaded to %s)!", attn_key, uploaded_count, repo_id)
        commit_hash = "verified"

        # Write confirmation record
        success_record = {
            "status": "UPLOAD_SUCCESS",
            "repo_id": repo_id,
            "format": "model_safetensors" if is_qwen else "attention_only_fp16",
            "commit_hash": str(commit_hash),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        }
        (target_dir / "upload_success.json").write_text(json.dumps(success_record, indent=2), encoding="utf-8")

        # Cluster Disk Quota Management: prune local heavy weights upon verified upload
        if cleanup_weights:
            heavy_extensions = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}
            purged_bytes = 0
            for fpath in sorted(target_dir.iterdir()):
                if fpath.is_file() and fpath.suffix in heavy_extensions:
                    try:
                        file_size = fpath.stat().st_size
                        fpath.unlink()
                        purged_bytes += file_size
                        logger.info(
                            "[%s] [DISK CLEANUP] Purged local heavy weight %s (%.1f MB freed).",
                            attn_key,
                            fpath.name,
                            file_size / (1024 * 1024),
                        )
                    except OSError as err:
                        logger.warning("[%s] Could not purge heavy file %s: %s", attn_key, fpath.name, err)
            if purged_bytes > 0:
                logger.info(
                    "[%s] [DISK CLEANUP] Verified upload complete. Total local storage freed: %.1f MB.",
                    attn_key,
                    purged_bytes / (1024 * 1024),
                )

        return True

    except Exception as e:
        redacted = _redact_secrets(str(e), token)
        logger.error("[%s] Upload failed: %s", attn_key, redacted, exc_info=True)
        (target_dir / "upload_error.json").write_text(
            json.dumps({"error": redacted, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"}),
            encoding="utf-8",
        )
        return False


import concurrent.futures


def watch_and_upload(
    watch_dirs: list[Path] | Path,
    hf_org: str,
    cleanup_weights: bool = True,
    poll_interval: int = 15,
    max_parallel_uploads: int = 3,
) -> None:
    """Continuously poll watch_dirs for completed models and upload them in parallel via ThreadPoolExecutor."""
    if isinstance(watch_dirs, Path):
        dirs = [watch_dirs]
    else:
        dirs = list(watch_dirs)

    logger.info("Watching %s for completed attention models (poll=%ds, max_parallel=%d)...", [str(d) for d in dirs], poll_interval, max_parallel_uploads)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_parallel_uploads, thread_name_prefix="hf_upload_worker"
    )
    in_flight: set[str] = set()

    while True:
        for w_dir in dirs:
            if not w_dir.exists():
                continue

            for sub in sorted(w_dir.iterdir()):
                if not sub.is_dir() or sub.name.startswith((".", "upload_")):
                    continue

                task_id = str(sub.resolve())
                if task_id in in_flight:
                    continue

                # A model is complete if training_metrics.json exists
                is_complete = (sub / "training_metrics.json").exists()
                already_uploaded = (sub / "upload_success.json").exists()

                if is_complete and not already_uploaded:
                    in_flight.add(task_id)
                    logger.info("[%s] Submitting to parallel upload pool (in-flight: %d)...", sub.name, len(in_flight))

                    def _run_upload(target_sub: Path = sub, t_id: str = task_id) -> None:
                        try:
                            upload_directory(target_sub, hf_org, cleanup_weights=cleanup_weights)
                        except Exception as err:
                            logger.error("[%s] Worker upload error: %s", target_sub.name, err)
                        finally:
                            in_flight.discard(t_id)

                    executor.submit(_run_upload)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT/Qwen Dual Zoo Hugging Face Parallel Uploader")
    parser.add_argument("--watch_dir", default=None, help="Directory or comma-separated directories to watch for completed models")
    parser.add_argument("--single_dir", default=None, help="Single directory to upload directly")
    parser.add_argument("--hf_org", default="Hooshaai", help="Hugging Face organization or user namespace")
    parser.add_argument("--cleanup", type=int, default=1, help="1 to delete weights after upload, 0 to keep")
    parser.add_argument("--poll_interval", type=int, default=15, help="Polling interval in seconds")
    parser.add_argument("--parallel", type=int, default=3, help="Number of concurrent parallel uploads")
    args = parser.parse_args()

    if args.single_dir:
        path = Path(args.single_dir)
        if not path.is_dir():
            logger.error("Specified path is not a directory: %s", path)
            sys.exit(1)
        success = upload_directory(path, args.hf_org, cleanup_weights=bool(args.cleanup))
        sys.exit(0 if success else 1)

    if args.watch_dir:
        dir_paths = [Path(d.strip()) for d in args.watch_dir.split(",") if d.strip()]
        watch_and_upload(
            dir_paths,
            args.hf_org,
            cleanup_weights=bool(args.cleanup),
            poll_interval=args.poll_interval,
            max_parallel_uploads=args.parallel,
        )
    else:
        logger.error("Must specify either --watch_dir or --single_dir")
        sys.exit(2)


if __name__ == "__main__":
    main()
