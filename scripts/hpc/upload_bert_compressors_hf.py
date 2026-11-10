#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dedicated Hugging Face Hub Uploader for Compressed + LoRA-Healed BERT Showcase Models.

Uploads complete serialized checkpoints (`pytorch_model.bin`, `config.json`,
`compression_summary.json`, `README.md`, and tokenizer assets) from
`results/bert_compressor_zoo_hf/<comp_key>/` directly to
`tahamajs/bert-comp-<slug>-sst2` WITHOUT stripping `pytorch_model.bin` pre-upload.
Optionally purges local `pytorch_model.bin` ONLY AFTER verified Hub commit.
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

os.environ["HF_ENDPOINT"] = "https://huggingface.co"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_bert_compressors")


def _resolve_repo_id(target_dir: Path, hf_org: str) -> str:
    key = target_dir.name
    slug = key.replace("_", "-")
    parent_name = target_dir.parent.name
    if "vlm" in parent_name or "llava" in parent_name:
        return f"{hf_org}/llava-qwen-0.5b-{slug}-scienceqa"
    if "tinyllama" in parent_name or "llama" in parent_name:
        return f"{hf_org}/tinyllama-1.1b-{slug}-wikitext2"
    return f"{hf_org}/bert-comp-{slug}-sst2"


def upload_compressed_dir(target_dir: Path, hf_org: str, cleanup_after: bool = True) -> bool:
    """Upload one compressed BERT, VLM, or TinyLlama checkpoint directory to Hugging Face Hub."""
    comp_key = target_dir.name
    repo_id = _resolve_repo_id(target_dir, hf_org)

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        logger.error("No Hugging Face token found.")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        commit_info = api.upload_folder(
            folder_path=str(target_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["upload_success.json", "upload_error.json", "*.tmp"],
            commit_message=f"Upload NeuralOps showcase checkpoint ({comp_key}) -> {repo_id}",
        )
        commit_hash = str(getattr(commit_info, "commit_id", getattr(commit_info, "oid", "verified")))
        logger.info("[%s] Uploaded cleanly to %s (commit=%s)", comp_key, repo_id, commit_hash)

        success = {
            "status": "UPLOAD_SUCCESS",
            "repo_id": repo_id,
            "commit_hash": commit_hash,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        (target_dir / "upload_success.json").write_text(json.dumps(success, indent=2), encoding="utf-8")

        if cleanup_after:
            for fname in ("pytorch_model.bin", "model.safetensors"):
                bin_path = target_dir / fname
                if bin_path.exists():
                    bin_path.unlink()
                    logger.info("[%s] Purged local %s post-upload.", comp_key, fname)
        return True
    except Exception as exc:
        logger.error("[%s] Upload failed: %s", comp_key, exc)
        (target_dir / "upload_error.json").write_text(
            json.dumps({"error": str(exc), "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}),
            encoding="utf-8",
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload BERT-Compressor, LLaVA-VLM & TinyLlama Zoos to HF Hub.")
    parser.add_argument(
        "--watch_dir",
        default="/home/tahamajlesi.ut/NeuralOps/results/bert_compressor_zoo_hf,/home/tahamajlesi.ut/NeuralOps/results/vlm_llava_scienceqa_zoo,/home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo",
    )
    parser.add_argument("--hf_org", default="tahamajs")
    parser.add_argument("--cleanup", type=int, default=1)
    parser.add_argument("--once", action="store_true", help="Upload all ready directories once and exit.")
    parser.add_argument("--poll_interval", type=int, default=15)
    args = parser.parse_args()

    watch_dirs = [Path(d.strip()) for d in args.watch_dir.split(",") if d.strip()]
    for w in watch_dirs:
        w.mkdir(parents=True, exist_ok=True)
    logger.info("Watching %s -> %s", [str(w) for w in watch_dirs], args.hf_org)

    while True:
        try:
            for watch in watch_dirs:
                if not watch.exists():
                    continue
                for sub in sorted(watch.iterdir()):
                    if not sub.is_dir() or sub.name.startswith((".", "upload_")):
                        continue
                    is_ready = (sub / "README.md").exists() and (
                        (sub / "compression_summary.json").exists() or (sub / "training_metrics.json").exists()
                    )
                    has_weights = (
                        (sub / "pytorch_model.bin").exists()
                        or (sub / "model.safetensors").exists()
                        or (sub / "attention_weights.safetensors").exists()
                    )
                    if is_ready and has_weights and not (sub / "upload_success.json").exists():
                        upload_compressed_dir(sub, args.hf_org, cleanup_after=bool(args.cleanup))
        except Exception as exc:
            logger.warning("Watcher loop error (will retry): %s", exc)
        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
