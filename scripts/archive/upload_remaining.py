"""Upload bs=64, 128, 256 (skip 32 which already succeeded)."""
import os, json, datetime, platform, subprocess
import torch
from huggingface_hub import HfApi, CommitOperationAdd

BASE_DIR    = r"D:\svd_project"
HF_ORG      = "TahaaaaM"
BATCH_SIZES = [64, 128, 256]

api = HfApi()

def get_env():
    try:
        nv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        nv = "unavailable"
    return {"python": platform.python_version(), "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": nv, "platform": platform.platform()}

def make_model_card(bs, metrics, env, repo_id):
    p = metrics["performance"]; e = metrics["efficiency"]; h = metrics["hardware"]
    return f"""---
language: en
tags: [svd, compression, attention, distilbert, sst2, a3-compressor]
datasets: [glue]
metrics: [accuracy, f1]
library_name: transformers
---
# SVD-Linear-Attention: DistilBERT-SST2 × A3 Compressor (batch_size={bs})

> **Status:** {p['status']} | **Accuracy:** {p['accuracy_pct']}% | **F1:** {p['f1_score']:.4f}

## Metrics

| Metric | Value |
|---|---|
| Accuracy | {p['accuracy_pct']}% |
| F1 (binary) | {p['f1_score']:.4f} |
| F1 (macro) | {p['f1_macro']:.4f} |
| Precision | {p['precision']:.4f} |
| Recall | {p['recall']:.4f} |
| Val Loss | {p['val_loss']:.4f} |
| Elapsed (s) | {p['elapsed_s']:.1f} |
| Peak VRAM (MB) | {h['peak_vram_mb']:.1f} |
| Tokens/sec | {h['tokens_per_sec']:.0f} |
| Compression Rate | {e['compression_rate']:.4f}x |
| Params Before | {e['total_params_before']:,} |
| Params After | {e['total_params_after']:,} |
| Compressed Layers | {e['num_compressed_layers']} |
| Avg Recon Error | {e['avg_reconstruction_error']:.4f} |

## Hyperparameters
- `batch_size`: {bs}  `rank`: 64  `max_error`: 0.50  `precision`: fp16

## Environment
```
Python: {env['python']}   PyTorch: {env['pytorch']}
CUDA:   {env['cuda']}     GPU:     {env['gpu']}
```

## Reproducibility
```python
from huggingface_hub import hf_hub_download
import torch
path = hf_hub_download(repo_id="{repo_id}", filename="finetuned_model.pt")
sd = torch.load(path, map_location="cpu")
print(len(sd), "keys loaded")
```

- GitHub: https://github.com/tahamajs/SVD_linear_Attention
- Generated: {datetime.datetime.now(datetime.UTC).isoformat()}
"""

env = get_env()
print("Env:", env)

for bs in BATCH_SIZES:
    result_dir   = os.path.join(BASE_DIR, "results", "bs_test", f"bs_{bs}")
    metrics_file = os.path.join(result_dir, "distilbert_sst2_a3_full_metrics.json")
    model_file   = os.path.join(result_dir, "finetuned_model.pt")
    summary_file = os.path.join(result_dir, "summary.csv")
    stats_file   = os.path.join(result_dir, "a3_stats.json")

    if not os.path.exists(metrics_file):
        print(f"[bs={bs}] SKIP — metrics not found"); continue

    with open(metrics_file) as f:
        metrics = json.load(f)

    status = metrics["performance"]["status"]
    acc    = metrics["performance"]["accuracy_pct"]
    print(f"\n[bs={bs}] Status={status}  Accuracy={acc}%")
    if status != "PASS":
        print(f"[bs={bs}] REJECTED"); continue

    repo_id = f"{HF_ORG}/svd-linear-attention-distilbert-sst2-a3-bs{bs}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"[bs={bs}] Repo: {repo_id}")

    ops = []
    if os.path.exists(model_file):
        ops.append(CommitOperationAdd("finetuned_model.pt",   model_file))
    ops.append(CommitOperationAdd("metrics.json",             metrics_file))
    if os.path.exists(summary_file):
        ops.append(CommitOperationAdd("summary.csv",          summary_file))
    if os.path.exists(stats_file):
        ops.append(CommitOperationAdd("a3_stats.json",        stats_file))
    env_txt = "\n".join(f"{k}: {v}" for k,v in env.items())
    ops.append(CommitOperationAdd("environment.txt",          env_txt.encode()))
    card = make_model_card(bs, metrics, env, repo_id)
    ops.append(CommitOperationAdd("README.md",                card.encode()))

    try:
        api.create_commit(
            repo_id=repo_id, repo_type="model", operations=ops,
            commit_message=f"benchmark(a3-bs{bs}): distilbert_sst2 {acc}% PASS",
        )
        files = [f.rfilename for f in api.list_repo_tree(repo_id=repo_id, repo_type="model")]
        print(f"[bs={bs}] UPLOADED -> https://huggingface.co/{repo_id}")
        print(f"[bs={bs}] Files: {files}")
    except Exception as e:
        print(f"[bs={bs}] FAILED: {e}")

print("\nDONE")
