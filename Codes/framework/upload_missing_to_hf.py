#!/usr/bin/env python3
"""
upload_missing_to_hf.py
Finds every single model and checkpoint across the entire project,
checks if it already exists on Hugging Face (Hooshaai),
and uploads all missing ones immediately.
"""
import os
import sys
import json
import glob
import datetime
import platform
import subprocess
import torch
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

HF_ORG = "Hooshaai"
api = HfApi()

def get_env():
    try:
        nv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        nv = "unavailable"
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", "N/A"),
        "gpu": nv,
        "platform": platform.platform()
    }

def get_existing_repos():
    print("Fetching existing repositories from Hugging Face...")
    try:
        repos = set(m.id for m in api.list_models(author=HF_ORG))
        print(f"Found {len(repos)} existing repositories under {HF_ORG}.")
        return repos
    except Exception as e:
        print(f"Error listing repos: {e}")
        return set()

def upload_target(repo_id, metrics_file, weights_file, summary_file, dataset_name, module_name, rank_str, env):
    import time
    for attempt in range(5):
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            break
        except Exception as err:
            err_str = str(err)
            if "429" in err_str or "rate limit" in err_str.lower():
                print(f"  [RATE-LIMIT 429] Waiting 65s for window reset (attempt {attempt+1}/5)...")
                time.sleep(65)
            else:
                print(f"  [WARN] Failed to create repo {repo_id}: {err}")
                return False
    else:
        print(f"  [ERROR] Exceeded retries for {repo_id}")
        return False

    perf_info = {}
    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                perf_info = d.get("performance", {})
        except Exception:
            pass

    acc = perf_info.get("accuracy_pct", "N/A")
    f1 = perf_info.get("f1_score", "N/A")
    status = perf_info.get("status", "PASS")

    card = f"""---
language: en
tags: [svd, linear-attention, transformer-compression, distilbert, {dataset_name}]
datasets: [{dataset_name}]
metrics: [accuracy, f1]
library_name: transformers
---
# SVD-Linear-Attention: DistilBERT-{dataset_name.upper()} × {module_name} ({rank_str})

- **Module**: `{module_name}`
- **Dataset**: `{dataset_name}`
- **Rank**: `{rank_str}`
- **Status**: `{status}`
- **Accuracy**: `{acc}%`
- **F1**: `{f1}`

Uploaded: {datetime.datetime.now(datetime.timezone.utc).isoformat()}
"""

    ops = [
        CommitOperationAdd("README.md", card.encode("utf-8")),
    ]
    if metrics_file and os.path.exists(metrics_file):
        ops.append(CommitOperationAdd("metrics.json", metrics_file))
    if weights_file and os.path.exists(weights_file):
        ops.append(CommitOperationAdd("weights.pt", weights_file))
    if summary_file and os.path.exists(summary_file):
        ops.append(CommitOperationAdd("summary.csv", summary_file))

    env_txt = "\n".join(f"{k}: {v}" for k, v in env.items())
    ops.append(CommitOperationAdd("environment.txt", env_txt.encode("utf-8")))

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=ops,
            commit_message=f"upload({module_name}): {dataset_name} ({rank_str})"
        )
        print(f"  [SUCCESS] -> https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Commit {repo_id}: {e}")
        return False

def scan_and_upload():
    env = get_env()
    existing_repos = get_existing_repos()
    uploaded_count = 0

    base_dir = r"D:\svd_project"

    # 1. SCAN RESULTS/RANK_ABLATION
    ablation_dir = os.path.join(base_dir, "results", "rank_ablation")
    if os.path.exists(ablation_dir):
        for stage in os.listdir(ablation_dir):
            stage_path = os.path.join(ablation_dir, stage)
            if not os.path.isdir(stage_path):
                continue
            parts = stage.split("_rank")
            ds = parts[0]
            r = parts[1] if len(parts) > 1 else "default"
            summary_file = os.path.join(stage_path, "summary.csv")

            for mf in glob.glob(os.path.join(stage_path, "*_full_metrics.json")):
                fname = os.path.basename(mf)
                mod = fname.replace("_full_metrics.json", "").split("_full_")[-1]
                if mod == fname:
                    mod = fname.replace("_full_metrics.json", "").split("_")[-1]

                repo_id = f"{HF_ORG}/svd-linear-attention-{ds}-{mod}-rank{r}"
                if repo_id in existing_repos:
                    continue

                weights_file = os.path.join(stage_path, "checkpoints", mod, "weights.pt")
                if not os.path.exists(weights_file):
                    weights_file = os.path.join(stage_path, "checkpoints", mod, "hf_export", "weights.pt")

                print(f"[MISSING ABLATION] Uploading {repo_id}...")
                if upload_target(repo_id, mf, weights_file, summary_file, ds, mod, f"rank{r}", env):
                    existing_repos.add(repo_id)
                    uploaded_count += 1

    # 2. SCAN RESULTS/UNIFIED_SWEEP
    sweep_dir = os.path.join(base_dir, "results", "unified_sweep")
    if os.path.exists(sweep_dir):
        for suite in os.listdir(sweep_dir):
            suite_path = os.path.join(sweep_dir, suite)
            if not os.path.isdir(suite_path):
                continue
            ds = "mnli" if "mnli" in suite else ("agnews" if "agnews" in suite else "sst2")
            summary_file = os.path.join(suite_path, "summary.csv")

            for mf in glob.glob(os.path.join(suite_path, "*_full_metrics.json")):
                fname = os.path.basename(mf)
                mod = fname.replace("_full_metrics.json", "").split("_full_")[-1]
                if mod == fname:
                    mod = fname.replace("_full_metrics.json", "").split("_")[-1]

                repo_id = f"{HF_ORG}/svd-linear-attention-{ds}-{mod}"
                if repo_id in existing_repos:
                    continue

                weights_file = os.path.join(suite_path, "checkpoints", mod, "weights.pt")
                if not os.path.exists(weights_file):
                    weights_file = os.path.join(suite_path, "checkpoints", mod, "hf_export", "weights.pt")

                print(f"[MISSING SWEEP] Uploading {repo_id}...")
                if upload_target(repo_id, mf, weights_file, summary_file, ds, mod, "default", env):
                    existing_repos.add(repo_id)
                    uploaded_count += 1

    # 3. SCAN RESULTS/{MODULE} DIRECT RUNS
    direct_results = os.path.join(base_dir, "results")
    for d in os.listdir(direct_results):
        mod_dir = os.path.join(direct_results, d)
        if not os.path.isdir(mod_dir) or d in ("rank_ablation", "unified_sweep", "bs_test", "benchmark"):
            continue
        # check for model file and metrics
        metrics = glob.glob(os.path.join(mod_dir, "*metrics*.json"))
        weights = glob.glob(os.path.join(mod_dir, "*.pt"))
        if metrics or weights:
            mf = metrics[0] if metrics else None
            wf = weights[0] if weights else None
            repo_id = f"{HF_ORG}/svd-linear-attention-standalone-{d}"
            if repo_id not in existing_repos:
                print(f"[MISSING STANDALONE] Uploading {repo_id}...")
                if upload_target(repo_id, mf, wf, None, "mixed", d, "standalone", env):
                    existing_repos.add(repo_id)
                    uploaded_count += 1

    print(f"\n==========================================")
    print(f"DONE! Uploaded {uploaded_count} previously unuploaded models to {HF_ORG}.")
    print(f"==========================================")

if __name__ == "__main__":
    scan_and_upload()
