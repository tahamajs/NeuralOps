#!/usr/bin/env python3
"""
upload_ablation_to_hf.py
Uploads all PASS/PASS_LOW rank ablation and sweep results to Hugging Face:
- Org: Hooshaai (or TahaaaaM fallback)
- Repos: svd-linear-attention-{dataset}-{module}-rank{rank} or central ablation dataset repo
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

def upload_single_result(metrics_path, rank, dataset_name, module_name, env):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    p = metrics.get("performance", {})
    e = metrics.get("efficiency", {})
    h = metrics.get("hardware", {})

    status = p.get("status", "UNKNOWN")
    acc = p.get("accuracy_pct", 0.0)
    if status not in ("PASS", "PASS_LOW"):
        print(f"  [SKIP] {module_name} status={status}")
        return False

    repo_id = f"{HF_ORG}/svd-linear-attention-{dataset_name}-{module_name}-rank{rank}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as err:
        print(f"  [WARN] Failed to create under {HF_ORG}, trying user namespace: {err}")
        repo_id = f"TahaaaaM/svd-linear-attention-{dataset_name}-{module_name}-rank{rank}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    result_dir = os.path.dirname(metrics_path)
    base_slug = f"distilbert_{dataset_name}_full_{module_name}" if "agnews" in dataset_name or "mnli" in dataset_name else f"distilbert_{dataset_name}_{module_name}"
    
    # Check for weights/checkpoint
    model_file = os.path.join(result_dir, "checkpoints", module_name, "weights.pt")
    if not os.path.exists(model_file):
        # Alternative locations
        cand = os.path.join(result_dir, "checkpoints", module_name, "model.pt")
        if os.path.exists(cand):
            model_file = cand

    card = f"""---
language: en
tags: [svd, linear-attention, transformer-compression, distilbert, {dataset_name}, rank-ablation]
datasets: [{dataset_name}]
metrics: [accuracy, f1]
library_name: transformers
---
# SVD-Linear-Attention: DistilBERT-{dataset_name.upper()} × {module_name} (Rank={rank})

- **Module**: `{module_name}`
- **SVD Rank**: `{rank}`
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
    if os.path.exists(model_file):
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
            commit_message=f"benchmark({module_name}-r{rank}): {dataset_name} acc={acc}% {status}"
        )
        print(f"  [UPLOADED] https://huggingface.co/{repo_id}")
        return True
    except Exception as exc:
        print(f"  [ERROR] Commit failed for {repo_id}: {exc}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=r"D:\svd_project\results\rank_ablation")
    parser.add_argument("--dataset", choices=["mnli", "sst2", "agnews", "all"], default="all")
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 64, 128, 192, 256])
    args = parser.parse_args()

    env = get_env()
    datasets = ["mnli", "sst2", "agnews"] if args.dataset == "all" else [args.dataset]

    total_uploaded = 0
    for ds in datasets:
        for r in args.ranks:
            sub_dir = os.path.join(args.base_dir, f"{ds}_rank{r}")
            if not os.path.isdir(sub_dir):
                continue
            print(f"\nProcessing {ds} rank={r} in {sub_dir}...")
            metric_files = glob.glob(os.path.join(sub_dir, "*_full_metrics.json"))
            for mf in metric_files:
                fname = os.path.basename(mf)
                # extract module name: e.g. distilbert_mnli_full_{module}_full_metrics.json
                parts = fname.replace("_full_metrics.json", "").split("_")
                # find module name
                mod = parts[-1]
                if len(parts) > 3:
                    # check if prefix is distilbert_{ds}_full or similar
                    mod = "_".join(parts[3:]) if "full" in parts else "_".join(parts[2:])
                
                if upload_single_result(mf, r, ds, mod, env):
                    total_uploaded += 1

    print(f"\nCompleted! Total uploaded to HuggingFace: {total_uploaded}")

if __name__ == "__main__":
    main()
