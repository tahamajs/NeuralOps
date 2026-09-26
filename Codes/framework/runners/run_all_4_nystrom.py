import subprocess
import os
import sys
import time
import json
import shutil
import pandas as pd
from pathlib import Path

# Ensure online access for automated Hugging Face uploads
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Switch to project directory if on Windows GPU server
if os.path.exists(r"PROJECT_ROOT"):
    HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
if os.path.exists(r"PROJECT_ROOT"):
    os.chdir(r"PROJECT_ROOT")
else:
    os.chdir(PROJECT_ROOT)
RESULTS_DIR = Path("results").resolve()

models = [
    {
        "name": "distilbert",
        "slug": "distilbert",
        "out_dir": r"results\unified_nystrom\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_nystrom\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,nystrom",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "16",
            "--rank", "32",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_nystrom\distilbert\summary.csv",
        "stats_json": r"results\unified_nystrom\distilbert\nystrom_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_nystrom\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_nystrom\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,nystrom",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "16",
            "--rank", "32",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_nystrom\roberta\summary.csv",
        "stats_json": r"results\unified_nystrom\roberta\nystrom_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_nystrom\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_nystrom\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,nystrom",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "16",
            "--rank", "32",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_nystrom\gpt2\summary.csv",
        "stats_json": r"results\unified_nystrom\gpt2\nystrom_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model NYSTROM Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,nystrom)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
parser.add_argument("--recovery_steps", type=int, default=None, help="Number of recovery fine-tuning steps")
parser.add_argument("--rank", type=int, default=None, help="Rank override")
parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank override")
parser.add_argument("positional_models", nargs="*", default=None, help="Positional model names for backwards compatibility")

args = parser.parse_args()

selected_models = []
if args.models:
    for item in args.models:
        for sub in item.split(","):
            if sub.strip():
                selected_models.append(sub.strip())
elif args.positional_models:
    for item in args.positional_models:
        for sub in item.split(","):
            if sub.strip():
                selected_models.append(sub.strip())

if selected_models:
    selected_set = set(selected_models)
    models = [m for m in models if m["name"] in selected_set]

def update_cmd_arg(cmd: list, flag: str, new_val: str):
    try:
        idx = cmd.index(flag)
        cmd[idx + 1] = str(new_val)
    except ValueError:
        cmd.extend([flag, str(new_val)])

for m in models:
    if args.modules:
        update_cmd_arg(m["cmd"], "--modules", args.modules)
    if args.batch_size is not None:
        update_cmd_arg(m["cmd"], "--batch_size", str(args.batch_size))
    if args.recovery_steps is not None:
        update_cmd_arg(m["cmd"], "--recovery_steps", str(args.recovery_steps))
    if args.rank is not None:
        update_cmd_arg(m["cmd"], "--rank", str(args.rank))
    if args.lora_rank is not None:
        update_cmd_arg(m["cmd"], "--lora_rank", str(args.lora_rank))

start_time = time.time()

for m in models:
    name = m["name"]
    out_dir = Path(m["out_dir"])
    cmd = m["cmd"]

    print(f"\n========================================================")
    print(f"Starting {name.upper()} Benchmark...")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================\n", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "console.log"

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    if proc.returncode != 0:
        print(f"\n[ERROR] Benchmark failed for {name} with returncode {proc.returncode}.", flush=True)
    else:
        print(f"\n[SUCCESS] Finished Benchmark for {name}.", flush=True)

print(f"\n========================================================")
print(f"All Requested Models Processed in {time.time() - start_time:.2f}s")
print(f"========================================================\n")

print("\n--- Summary Results ---")
for m in models:
    name = m["name"]
    summary_path = Path(m["summary_csv"])
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            print(f"\nSummary for {name}:")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Could not read summary CSV for {name}: {e}")
    else:
        print(f"No summary CSV found for {name} at {summary_path}")

print(f"\n========================================================")
print(f"Starting Quality Gate & Automated Hugging Face Export...")
print(f"========================================================\n", flush=True)

try:
    from huggingface_hub import HfApi
    api = HfApi()
except ImportError:
    api = None
    print("[WARN] huggingface_hub is not installed. Automated HF upload will be skipped.", flush=True)

hf_org = "Hooshaai"
pass_count = 0
uploaded_repos = []

for m in models:
    name = m["name"]
    slug = m["slug"]
    out_dir = Path(m["out_dir"])
    summary_path = Path(m["summary_csv"])
    weights_dir = out_dir / "weights"
    hf_export_dir = out_dir / "hf_export"

    if not summary_path.exists():
        print(f"[Quality Gate: REJECTED] {name} summary CSV does not exist. Skipping export and cleaning up.", flush=True)
        if weights_dir.exists():
            shutil.rmtree(weights_dir, ignore_errors=True)
        continue

    row = {}
    try:
        df = pd.read_csv(summary_path)
        if not df.empty:
            if "status" in df.columns:
                cand = df[df["module"] != "full"] if "module" in df.columns else df
                if not cand.empty:
                    row = cand.iloc[-1].to_dict()
                else:
                    row = df.iloc[-1].to_dict()
            else:
                row = df.iloc[-1].to_dict()
    except Exception as e:
        print(f"Error reading summary for {name}: {e}", flush=True)

    status = str(row.get("status", "FAIL"))
    try:
        acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0.0)))
    except Exception:
        acc = 0.0

    is_successful = (status == "PASS") and (acc >= 56.0)

    if not is_successful:
        print(f"[Quality Gate: REJECTED] {name} did not meet criteria (status={status}, acc={acc:.2f}% < 56.0%).", flush=True)
        if weights_dir.exists():
            shutil.rmtree(weights_dir, ignore_errors=True)
        continue

    pass_count += 1
    print(f"[Quality Gate: PASSED] {name} passed criteria (status={status}, acc={acc:.2f}% >= 56.0%). Packaging for export...", flush=True)

    hf_export_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    weights_candidates = (
        list(checkpoints_dir.glob("**/weights.pt"))
        + list(weights_dir.glob("**/weights.pt"))
        + list(out_dir.glob("**/weights.pt"))
    )
    if weights_candidates:
        src_weights = weights_candidates[0]
        dst_weights = hf_export_dir / "weights.pt"
        if src_weights.resolve() != dst_weights.resolve():
            shutil.copy2(src_weights, dst_weights)
        # Also copy config and tokenizer files from the checkpoint directory
        src_parent = src_weights.parent
        for f_extra in src_parent.glob("*"):
            if f_extra.is_file() and f_extra.name not in ("weights.pt", "pytorch_model.bin"):
                shutil.copy2(f_extra, hf_export_dir / f_extra.name)

    # Save complete publication-ready metrics
    metrics_payload = {
        "model_name": name,
        "module": "nystrom",
        "status": status,
        "accuracy_pct": acc,
        "f1": float(row.get("f1", row.get("mean_f1", 0.0))),
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "eval_loss": float(row.get("eval_loss", 0.0)),
        "params_total": row.get("params_total", None),
        "params_active": row.get("params_active", None),
        "param_reduction_pct": row.get("param_reduction_pct", None),
        "peak_vram_mb": row.get("peak_vram_mb", None),
        "tokens_per_sec": row.get("tokens_per_sec", None),
        "elapsed_seconds": row.get("elapsed_seconds", None),
        "full_metrics": row,
    }
    with open(hf_export_dir / "metrics.json", "w", encoding="utf-8") as mf:
        json.dump(metrics_payload, mf, indent=2)

    # Comprehensive, publication-ready model card
    card_content = f"""---
tags:
- svd-linear-attention
- model-compression
- {slug}
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: {name.upper()} ({slug})

Validated factorized/adapted checkpoint of **{name}** using **{slug}** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `{acc:.2f}%`
- **F1 Score (Binary)**: `{row.get('f1_score', row.get('f1', 'N/A'))}`
- **F1 Macro**: `{row.get('f1_macro', 'N/A')}`
- **Precision**: `{row.get('precision', 'N/A')}`
- **Recall**: `{row.get('recall', 'N/A')}`
- **Validation Loss**: `{row.get('val_loss', 'N/A')}`
- **Quality Gate Status**: `{status}` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `{row.get('compression_rate', '1.00')}x`
- **FLOPs Reduction**: `{row.get('flops_reduction_pct', '0.0')}%`
- **Total Parameters Before**: `{row.get('total_params_before', 'N/A')}`
- **Total Parameters After**: `{row.get('total_params_after', 'N/A')}`
- **Trainable Recovery Params**: `{row.get('trainable_recovery_params', 'N/A')}`
- **Compressed / Injected Layers**: `{row.get('num_compressed_layers', row.get('num_injected_attention_layers', 'N/A'))}`
- **Peak VRAM**: `{row.get('peak_vram_mb', 'N/A')} MB`
- **Throughput**: `{row.get('tokens_per_sec', 'N/A')} tokens/sec`
- **Evaluation Time**: `{row.get('elapsed_s', 'N/A')} s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
"""
    with open(hf_export_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(card_content)

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-nystrom"
    if api is not None:
        try:
            print(f"Uploading to Hugging Face: {repo_id}...", flush=True)
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(hf_export_dir),
                repo_id=repo_id,
                commit_message=f"Upload hardened NYSTROM model weights ({name}) - Acc: {acc:.2f}%",
            )
            print(f"Successfully uploaded {name} to {repo_id}", flush=True)
            uploaded_repos.append(repo_id)
        except Exception as e:
            print(f"Upload failed for {repo_id}: {e}", flush=True)

print(f"\nExport Complete: {pass_count}/{len(models)} passed quality gate.", flush=True)
