"""
Upload all 4 batch-size sweep results (bs=32,64,128,256) to HuggingFace.
Repo pattern: TahaaaaM/svd-linear-attention-distilbert-sst2-a3-bs{BS}
"""

import os
import json
import datetime
import platform
import subprocess
import torch
from huggingface_hub import HfApi, CommitOperationAdd

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\svd_project"
MODEL_NAME = "distilbert_sst2"
MODULE     = "a3"
HF_ORG     = "TahaaaaM"
BATCH_SIZES = [32, 64, 128, 256]

api = HfApi()
HF_USER = api.whoami()["name"]

def get_env():
    try:
        nv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        nv = "unavailable"
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": nv,
        "platform": platform.platform(),
    }

def make_model_card(bs, metrics, env, repo_id):
    perf = metrics["performance"]
    eff  = metrics["efficiency"]
    hw   = metrics["hardware"]
    return f"""---
language: en
tags:
  - svd
  - compression
  - attention
  - distilbert
  - sst2
  - a3-compressor
datasets:
  - glue
metrics:
  - accuracy
  - f1
library_name: transformers
---

# SVD-Linear-Attention: DistilBERT-SST2 × A3 Compressor (batch_size={bs})

> **Status:** {perf['status']} | **Accuracy:** {perf['accuracy_pct']}% | **F1:** {perf['f1_score']:.4f}

## Description

A3 (Activation-Aware Factorized SVD) compressor applied to **DistilBERT-SST2**.
This variant was benchmarked with `batch_size={bs}` during the quality sweep.

## Metrics

| Metric | Value |
|---|---|
| Accuracy | {perf['accuracy_pct']}% |
| F1 (binary) | {perf['f1_score']:.4f} |
| F1 (macro) | {perf['f1_macro']:.4f} |
| Precision | {perf['precision']:.4f} |
| Recall | {perf['recall']:.4f} |
| Val Loss | {perf['val_loss']:.4f} |
| Elapsed (s) | {perf['elapsed_s']:.1f} |
| Peak VRAM (MB) | {hw['peak_vram_mb']:.1f} |
| Tokens/sec | {hw['tokens_per_sec']:.0f} |
| Compression Rate | {eff['compression_rate']:.4f}x |
| Params Before | {eff['total_params_before']:,} |
| Params After | {eff['total_params_after']:,} |
| Compressed Layers | {eff['num_compressed_layers']} |
| Avg Recon Error | {eff['avg_reconstruction_error']:.4f} |

## Hyperparameters

- `batch_size`: {bs}
- `rank`: 64
- `max_error`: 0.50
- `recovery_steps`: 50 (head fine-tune 200 steps)
- `precision`: fp16
- `optimizer`: adamw (lr=2e-4)

## Environment

```
Python:  {env['python']}
PyTorch: {env['pytorch']}
CUDA:    {env['cuda']}
GPU:     {env['gpu']}
```

## Reproducibility

```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(repo_id="{repo_id}", filename="finetuned_model.pt")
state_dict = torch.load(model_path, map_location="cpu")
print("Loaded state dict with", len(state_dict), "keys")
```

## Links

- **GitHub:** https://github.com/tahamajs/SVD_linear_Attention
- **Base model:** distilbert-base-uncased-finetuned-sst-2-english

---
*Generated: {datetime.datetime.utcnow().isoformat()}Z*
"""

env = get_env()
print("Environment:", env)

for bs in BATCH_SIZES:
    result_dir = os.path.join(BASE_DIR, "results", "bs_test", f"bs_{bs}")
    metrics_file = os.path.join(result_dir, f"distilbert_sst2_a3_full_metrics.json")
    model_file   = os.path.join(result_dir, "finetuned_model.pt")
    summary_file = os.path.join(result_dir, "summary.csv")
    stats_file   = os.path.join(result_dir, "a3_stats.json")

    if not os.path.exists(metrics_file):
        print(f"[bs={bs}] Skipping — metrics file not found")
        continue

    with open(metrics_file) as f:
        metrics = json.load(f)

    status = metrics["performance"]["status"]
    acc    = metrics["performance"]["accuracy_pct"]
    print(f"\n[bs={bs}] Status={status} Accuracy={acc}%")

    if status != "PASS":
        print(f"[bs={bs}] REJECTED (not PASS) — skipping HF upload")
        continue

    # ── repo ──────────────────────────────────────────────────────────────
    repo_id = f"{HF_ORG}/svd-linear-attention-distilbert-sst2-a3-bs{bs}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        print(f"[bs={bs}] Repo ready: {repo_id}")
    except Exception as e:
        print(f"[bs={bs}] Repo creation error (may already exist): {e}")

    # ── build commit operations ────────────────────────────────────────────
    ops = []

    # Model weights
    if os.path.exists(model_file):
        ops.append(CommitOperationAdd(
            path_in_repo="finetuned_model.pt",
            path_or_fileobj=model_file,
        ))

    # Metrics JSON
    ops.append(CommitOperationAdd(
        path_in_repo="metrics.json",
        path_or_fileobj=metrics_file,
    ))

    # Summary CSV
    if os.path.exists(summary_file):
        ops.append(CommitOperationAdd(
            path_in_repo="summary.csv",
            path_or_fileobj=summary_file,
        ))

    # A3 stats
    if os.path.exists(stats_file):
        ops.append(CommitOperationAdd(
            path_in_repo="a3_stats.json",
            path_or_fileobj=stats_file,
        ))

    # Environment file
    env_txt = "\n".join(f"{k}: {v}" for k, v in env.items())
    env_bytes = env_txt.encode()
    ops.append(CommitOperationAdd(
        path_in_repo="environment.txt",
        path_or_fileobj=env_bytes,
    ))

    # Model card
    card = make_model_card(bs, metrics, env, repo_id)
    ops.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=card.encode(),
    ))

    # ── commit ────────────────────────────────────────────────────────────
    try:
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=ops,
            commit_message=f"benchmark(a3-bs{bs}): distilbert_sst2 {acc}% acc PASS",
        )
        # Verify upload
        files = [f.rfilename for f in api.list_repo_tree(repo_id=repo_id, repo_type="model")]
        print(f"[bs={bs}] ✓ Uploaded to https://huggingface.co/{repo_id}")
        print(f"[bs={bs}]   Files: {files}")
    except Exception as e:
        print(f"[bs={bs}] Upload FAILED: {e}")

print("\nDONE")
