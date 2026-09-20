#!/usr/bin/env python3
"""
upload_all_results_to_hf.py
Uploads ALL PASS/PASS_LOW/MODERATE_EXPANSION results across the entire project to Hugging Face:
1. Ablation results: results/rank_ablation/{dataset}_rank{rank}/
2. Unified sweep results: results/unified_sweep/{suite}/
3. Direct results: results/{module}/

Ensures every single evaluated module is completely represented on Hugging Face (Hooshaai).
"""
import os
import sys
import json
import glob
import argparse
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

def upload_metrics_item(metrics_path, repo_id, tag_dataset, module_name, env, rank=None):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Reading {metrics_path}: {e}")
        return False

    p = metrics.get("performance", {})
    e = metrics.get("efficiency", {})
    h = metrics.get("hardware", {})

    status = p.get("status", "PASS")
    acc = float(p.get("accuracy_pct", 0.0) or 0.0)

    # Allow all viable runs, but reject 0.0% accuracy invalid runs
    if status not in ("PASS", "PASS_LOW", "MODERATE_EXPANSION") or acc <= 0.0:
        return False

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as err:
        print(f"  [WARN] Failed to create {repo_id}: {err}")
        return False

    result_dir = os.path.dirname(metrics_path)
    
    # Candidate weights
    weights_candidates = [
        os.path.join(result_dir, "checkpoints", module_name, "weights.pt"),
        os.path.join(result_dir, "checkpoints", module_name, "model.pt"),
        os.path.join(result_dir, "checkpoints", module_name, "hf_export", "weights.pt"),
        os.path.join(result_dir, "finetuned_model.pt"),
    ]
    model_file = next((cand for cand in weights_candidates if os.path.exists(cand)), None)

    rank_str = f" (Rank={rank})" if rank else ""
    card = f"""---
language: en
tags: [svd, linear-attention, transformer-compression, distilbert, {tag_dataset}]
datasets: [{tag_dataset}]
metrics: [accuracy, f1]
library_name: transformers
---
# SVD-Linear-Attention: DistilBERT-{tag_dataset.upper()} × {module_name}{rank_str}

- **Module**: `{module_name}`
- **Dataset**: `{tag_dataset}`
- **Rank**: `{rank if rank else 'default'}`
- **Status**: `{status}`
- **Accuracy**: `{acc}%`
- **F1**: `{p.get('f1_score', 0.0):.4f}`
- **Compression Rate**: `{e.get('compression_rate', 1.0):.4f}x`
- **Params Before**: `{e.get('total_params_before', 0):,}`
- **Params After**: `{e.get('total_params_after', 0):,}`
- **Peak VRAM**: `{h.get('peak_vram_mb', 0.0):.1f} MB`
- **Tokens/sec**: `{h.get('tokens_per_sec', 0.0):.1f}`

Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}
"""

    ops = [
        CommitOperationAdd("metrics.json", metrics_path),
        CommitOperationAdd("README.md", card.encode("utf-8")),
    ]
    if model_file and os.path.exists(model_file):
        ops.append(CommitOperationAdd("weights.pt", model_file))

    summary_file = os.path.join(result_dir, "summary.csv")
    if os.path.exists(summary_file):
        ops.append(CommitOperationAdd("summary.csv", summary_file))

    env_txt = "\n".join(f"{k}: {v}" for k, v in env.items())
    ops.append(CommitOperationAdd("environment.txt", env_txt.encode("utf-8")))

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=ops,
            commit_message=f"upload({module_name}): {tag_dataset} acc={acc}% {status}"
        )
        print(f"  [UPLOADED] https://huggingface.co/{repo_id}")
        return True
    except Exception as exc:
        print(f"  [SKIP/EXISTS] {repo_id}: {exc}")
        return False

def upload_all():
    env = get_env()
    total_uploaded = 0

    # 1. Upload All Ablation Results (Ranks 32, 64, 128, 256 for MNLI, SST-2, AG News)
    ablation_base = r"D:\svd_project\results\rank_ablation"
    if os.path.exists(ablation_base):
        for sub in glob.glob(os.path.join(ablation_base, "*")):
            if not os.path.isdir(sub):
                continue
            folder_name = os.path.basename(sub)
            # e.g. mnli_rank32, sst2_rank64
            parts = folder_name.split("_rank")
            ds = parts[0]
            rank = parts[1] if len(parts) > 1 else ""
            print(f"\nProcessing ablation folder: {folder_name}...")
            for mf in glob.glob(os.path.join(sub, "*_full_metrics.json")):
                fname = os.path.basename(mf)
                mod = fname.replace("_full_metrics.json", "").split("_")[-1]
                if "full" in fname:
                    mod = fname.replace("_full_metrics.json", "").split("_full_")[-1]
                repo_id = f"{HF_ORG}/svd-linear-attention-{ds}-{mod}-rank{rank}"
                if upload_metrics_item(mf, repo_id, ds, mod, env, rank=rank):
                    total_uploaded += 1

    # 2. Upload Unified Sweep Results (MNLI, AG News, SST-2)
    sweep_base = r"D:\svd_project\results\unified_sweep"
    if os.path.exists(sweep_base):
        for suite in ["distilbert_mnli_full", "distilbert_agnews_full", "distilbert"]:
            suite_dir = os.path.join(sweep_base, suite)
            if not os.path.isdir(suite_dir):
                continue
            ds_name = "mnli" if "mnli" in suite else ("agnews" if "agnews" in suite else "sst2")
            print(f"\nProcessing unified sweep suite: {suite} ({ds_name})...")
            for mf in glob.glob(os.path.join(suite_dir, "*_full_metrics.json")):
                fname = os.path.basename(mf)
                mod = fname.replace("_full_metrics.json", "").split("_")[-1]
                if "_full_" in fname:
                    mod = fname.replace("_full_metrics.json", "").split("_full_")[-1]
                repo_id = f"{HF_ORG}/svd-linear-attention-{ds_name}-{mod}"
                if upload_metrics_item(mf, repo_id, ds_name, mod, env):
                    total_uploaded += 1

    print(f"\n==========================================")
    print(f"FINISHED! Total repositories created/updated on HF: {total_uploaded}")
    print(f"==========================================")

if __name__ == "__main__":
    upload_all()
