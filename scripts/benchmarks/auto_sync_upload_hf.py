#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated Continuous Hugging Face Sync & Upload Daemon.

Continuously monitors trained models on the Sharif HPC cluster (results/bert_attention_zoo),
transfers them via high-speed rsync, uploads them directly to Hugging Face Hub (tahamajs/bert-<slug>-sst2),
and cleans up heavy weights to preserve cluster disk quotas.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["NO_PROXY"] = "*"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hf_auto_syncer")


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    return token or ""


def run_ssh(cmd: str) -> str:
    res = subprocess.run(["ssh", "hpc", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout.strip()


def sync_and_upload_loop(
    remote_base: str = "/home/tahamajlesi.ut/NeuralOps/results/bert_attention_zoo",
    hf_org: str = "tahamajs",
    poll_interval: int = 20,
    temp_dir: str = "/tmp/bert_zoo_sync",
) -> None:
    from huggingface_hub import HfApi

    token = get_hf_token()
    api = HfApi(token=token)
    logger.info("HF Auto-Sync Daemon active. Watching remote %s on hpc (poll=%ds)...", remote_base, poll_interval)

    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    while True:
        try:
            # Query completed subdirectories on cluster
            out = run_ssh(f"ls -d {remote_base}/* 2>/dev/null")
            lines = [l.strip() for l in out.splitlines() if l.strip()]

            for r_dir in lines:
                name = Path(r_dir).name
                if name.startswith((".", "upload_")) or name in ("baseline_ppl.json", "metrics.jsonl", "registry_keys.json", "skipped_keys.json", "teacher_sst2_metrics.json"):
                    continue

                # Check if trained and if not already uploaded
                check_cmd = f"[ -f {r_dir}/training_metrics.json ] && echo 'TRAINED'; [ -f {r_dir}/upload_success.json ] && echo 'UPLOADED'"
                status = run_ssh(check_cmd).splitlines()

                if "TRAINED" in status and "UPLOADED" not in status:
                    slug = name.replace("_", "-")
                    repo_id = f"{hf_org}/bert-{slug}-sst2"
                    logger.info("[%s] New completed model found! Syncing and uploading to %s...", name, repo_id)

                    local_dest = Path(temp_dir) / name
                    local_dest.mkdir(parents=True, exist_ok=True)

                    # Rsync directory from cluster
                    sync_cmd = [
                        "rsync",
                        "-avz",
                        f"hpc:{r_dir}/",
                        f"{local_dest}/",
                    ]
                    subprocess.run(sync_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                    # Create repo and upload folder
                    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
                    commit_info = api.upload_folder(
                        folder_path=str(local_dest),
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Pretrained BERT-{slug} distilled on GLUE SST-2 (NeuralOps Attention Zoo)",
                    )
                    logger.info("[%s] Upload successful! Remote commit: %s", name, commit_info)

                    # Mark remote success atomically via POSIX mv
                    success_record = {
                        "status": "UPLOAD_SUCCESS",
                        "repo_id": repo_id,
                        "commit_hash": str(getattr(commit_info, "commit_id", "success")),
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    }
                    record_json = json.dumps(success_record)
                    run_ssh(f"echo '{record_json}' > {r_dir}/upload_success.json.tmp && mv {r_dir}/upload_success.json.tmp {r_dir}/upload_success.json")

                    # Purge remote heavy weights
                    run_ssh(f"rm -f {r_dir}/pytorch_model.bin {r_dir}/model.safetensors {r_dir}/optimizer.pt")
                    logger.info("[%s] Remote heavy weights purged to preserve cluster quota.", name)

                    # Clean local temp
                    subprocess.run(["rm", "-rf", str(local_dest)])

        except Exception as e:
            logger.error("Error in sync & upload cycle: %s", e)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous HF Sync & Upload Daemon")
    parser.add_argument("--remote_base", default="/home/tahamajlesi.ut/NeuralOps/results/bert_attention_zoo")
    parser.add_argument("--hf_org", default="tahamajs")
    parser.add_argument("--poll_interval", type=int, default=20)
    args = parser.parse_args()

    sync_and_upload_loop(
        remote_base=args.remote_base,
        hf_org=args.hf_org,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
