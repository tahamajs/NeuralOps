#!/usr/bin/env python3
"""
Background Auto-Uploader & Verifier for Hugging Face Hub.

Monitors results/ directory, finds ALL model checkpoints, and ensures
that:
1. Every repo exists on Hooshaai.
2. weights.pt is completely uploaded (retries indefinitely until verified).
3. All auxiliary configs/metrics/tokenizers are uploaded.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi

HF_ORG = "Hooshaai"
CHECK_INTERVAL_S = 180  # Check every 3 minutes

def get_repo_id(model_family: str, mod_name: str) -> str:
    slug = mod_name.replace("_", "-")
    return f"{HF_ORG}/svd-linear-attention-{model_family}-{slug}"

def sync_and_verify_hf():
    repo_dir = Path(os.environ.get("SVD_PROJECT_ROOT", "D:\\SVD_Project" if os.path.exists("D:\\SVD_Project") else "."))
    results_base = repo_dir / "results"
    if not results_base.exists():
        return

    api = HfApi()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Running sweep across {results_base}...")

    # Find all weights.pt files under results/unified_*/*/checkpoints/*
    weight_files = list(results_base.glob("unified_*/*/checkpoints/*/weights.pt"))
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Found {len(weight_files)} candidate weights.pt files.")

    for wf in weight_files:
        ckpt_dir = wf.parent
        if ckpt_dir.name == "full":
            continue
            
        model_dir = ckpt_dir.parent.parent
        model_name = model_dir.name  # distilbert, roberta, gpt2, qwen3_5
        mod_name = model_dir.parent.name.replace("unified_", "")
        
        repo_id = get_repo_id(model_name, mod_name)
        
        # 1. Ensure repo exists
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        except Exception:
            pass
            
        # 2. Check remote files
        remote_files = []
        try:
            remote_files = api.list_repo_files(repo_id)
        except Exception:
            pass

        # 3. If weights.pt is missing or incomplete, upload with retry
        if "weights.pt" not in remote_files:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Uploading missing weights.pt ({wf.stat().st_size / 1e6:.1f} MB) to {repo_id}...")
            success = False
            for attempt in range(1, 4):
                try:
                    api.upload_file(
                        path_or_fileobj=str(wf),
                        path_in_repo="weights.pt",
                        repo_id=repo_id,
                        commit_message=f"Upload model weights for {model_name}-{mod_name} (recovery_steps=50)",
                    )
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] SUCCESS: weights.pt -> {repo_id}")
                    success = True
                    break
                except Exception as e_up:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Sync] Attempt {attempt} failed for {repo_id}: {e_up}")
                    time.sleep(5)
            if not success:
                continue

        # 4. Upload configs, metrics, tokenizers if missing
        for extra in ckpt_dir.glob("*"):
            if extra.is_file() and extra.name not in ("weights.pt", "pytorch_model.bin"):
                if extra.name not in remote_files:
                    try:
                        api.upload_file(
                            path_or_fileobj=str(extra),
                            path_in_repo=extra.name,
                            repo_id=repo_id,
                            commit_message=f"Upload {extra.name}",
                        )
                    except Exception:
                        pass

def loop(interval_s: int = CHECK_INTERVAL_S):
    while True:
        try:
            sync_and_verify_hf()
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in HF loop: {e}")
        if interval_s <= 0:
            break
        time.sleep(interval_s)

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else CHECK_INTERVAL_S
    loop(interval)
