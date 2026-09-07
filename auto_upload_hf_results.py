#!/usr/bin/env python3
"""
Background Auto-Uploader for Hugging Face Hub.

Periodically monitors the results/ directory for newly completed benchmarks
and automatically creates/updates Hugging Face repositories under the Hooshaai
organization with:
1. Model weights (weights.pt)
2. Tokenizer and config files
3. Evaluation metrics and model cards
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi

HF_ORG = "Hooshaai"
CHECK_INTERVAL_S = 300  # Check every 5 minutes

def get_repo_id(model_family: str, mod_name: str) -> str:
    slug = mod_name.replace("_", "-")
    return f"{HF_ORG}/svd-linear-attention-{model_family}-{slug}"

def auto_upload_hf_loop(interval_s: int = CHECK_INTERVAL_S):
    repo_dir = Path(os.environ.get("SVD_PROJECT_ROOT", "D:\\SVD_Project" if os.path.exists("D:\\SVD_Project") else "."))
    results_base = repo_dir / "results"
    
    api = HfApi()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auto-Uploader started for Hugging Face org '{HF_ORG}'. Checking every {interval_s}s...")

    uploaded_record = set()

    while True:
        try:
            if results_base.exists():
                for unified_dir in sorted(results_base.glob("unified_*")):
                    mod_name = unified_dir.name.replace("unified_", "")
                    
                    for model_dir in unified_dir.glob("*"):
                        if not model_dir.is_dir():
                            continue
                        model_name = model_dir.name  # distilbert, roberta, gpt2, qwen3_5
                        
                        ckpt_candidates = list((model_dir / "checkpoints").glob("*")) if (model_dir / "checkpoints").exists() else []
                        for ckpt_dir in ckpt_candidates:
                            if ckpt_dir.name == "full":
                                continue
                            
                            weights_file = ckpt_dir / "weights.pt"
                            if not weights_file.exists():
                                continue
                            
                            cache_key = f"{model_name}:{mod_name}:{ckpt_dir.name}"
                            if cache_key in uploaded_record:
                                continue
                            
                            repo_id = get_repo_id(model_name, mod_name)
                            
                            try:
                                api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
                            except Exception as e_cr:
                                pass
                            
                            try:
                                remote_files = api.list_repo_files(repo_id)
                            except Exception:
                                remote_files = []
                                
                            if "weights.pt" in remote_files:
                                uploaded_record.add(cache_key)
                                continue

                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Upload] Uploading {weights_file} to {repo_id}...")
                            try:
                                api.upload_file(
                                    path_or_fileobj=str(weights_file),
                                    path_in_repo="weights.pt",
                                    repo_id=repo_id,
                                    commit_message=f"Upload model weights for {model_name}-{mod_name} (recovery_steps=50)",
                                )
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Upload] SUCCESS: weights.pt -> {repo_id}")
                                
                                for extra in ckpt_dir.glob("*"):
                                    if extra.is_file() and extra.name not in ("weights.pt", "pytorch_model.bin"):
                                        try:
                                            api.upload_file(
                                                path_or_fileobj=str(extra),
                                                path_in_repo=extra.name,
                                                repo_id=repo_id,
                                                commit_message=f"Upload config/metrics {extra.name}",
                                            )
                                        except Exception:
                                            pass
                                            
                                uploaded_record.add(cache_key)
                            except Exception as up_err:
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HF-Upload] Failed {repo_id}: {up_err}")

        except Exception as loop_err:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error in upload loop: {loop_err}")

        if interval_s <= 0:
            break
        time.sleep(interval_s)

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else CHECK_INTERVAL_S
    auto_upload_hf_loop(interval)
