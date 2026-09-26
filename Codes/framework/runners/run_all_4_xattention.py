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
        "out_dir": r"results\unified_xattention\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_xattention\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,xattention",
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
        "summary_csv": r"results\unified_xattention\distilbert\summary.csv",
        "stats_json": r"results\unified_xattention\distilbert\xattention_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_xattention\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_xattention\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,xattention",
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
        "summary_csv": r"results\unified_xattention\roberta\summary.csv",
        "stats_json": r"results\unified_xattention\roberta\xattention_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_xattention\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_xattention\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,xattention",
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
        "summary_csv": r"results\unified_xattention\gpt2\summary.csv",
        "stats_json": r"results\unified_xattention\gpt2\xattention_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model Block Sparse Attention (XAttention) Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,xattention)")
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
    if args.modules is not None:
        update_cmd_arg(m["cmd"], "--modules", args.modules)
    if args.batch_size is not None:
        update_cmd_arg(m["cmd"], "--batch_size", str(args.batch_size))
    if args.recovery_steps is not None:
        update_cmd_arg(m["cmd"], "--recovery_steps", str(args.recovery_steps))
    if args.rank is not None:
        update_cmd_arg(m["cmd"], "--rank", str(args.rank))
    if args.lora_rank is not None:
        update_cmd_arg(m["cmd"], "--lora_rank", str(args.lora_rank))

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED XATTENTION BENCHMARK ON GPU SERVER", flush=True)
print(f"Running models: {[m['name'] for m in models]}", flush=True)
print("=" * 80, flush=True)

final_stats = []

for m in models:
    print(f"\n{'='*30} RUNNING MODEL: {m['name']} {'='*30}", flush=True)
    t0 = time.time()
    res = subprocess.run(m["cmd"])
    elapsed = time.time() - t0

    acc = None
    baseline_acc = None
    status = "UNKNOWN" if res.returncode == 0 else f"EXIT_{res.returncode}"
    num_compressed_layers = "N/A"
    f1 = None
    comp_rate = None
    peak_vram = None

    csv_path = Path(m["summary_csv"])
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                col = "accuracy_pct" if "accuracy_pct" in df.columns else ("mean_accuracy" if "mean_accuracy" in df.columns else None)
                if col:
                    row_base = df[df["module_name"] == "full"]
                    if not row_base.empty:
                        baseline_acc = float(row_base[col].values[-1])
                    row = df[df["module_name"] == "xattention"]
                    if not row.empty:
                        acc = float(row[col].values[-1])
                        status = str(row["status"].values[-1])
                        if "f1_score" in row.columns and not pd.isna(row["f1_score"].values[-1]):
                            try: f1 = float(row["f1_score"].values[-1])
                            except Exception: pass
                        if "compression_rate" in row.columns and not pd.isna(row["compression_rate"].values[-1]):
                            try: comp_rate = float(row["compression_rate"].values[-1])
                            except Exception: pass
                        if "peak_vram_mb" in row.columns and not pd.isna(row["peak_vram_mb"].values[-1]):
                            try: peak_vram = float(row["peak_vram_mb"].values[-1])
                            except Exception: pass
                        if "num_compressed_layers" in row.columns and not pd.isna(row["num_compressed_layers"].values[-1]):
                            num_compressed_layers = str(row["num_compressed_layers"].values[-1])
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}", flush=True)

    stats_path = Path(m.get("stats_json", ""))
    if stats_path.exists() and num_compressed_layers == "N/A":
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                num_compressed_layers = s_data.get("num_compressed_layers", "N/A")
        except Exception as e:
            print(f"Error reading stats JSON for {m['name']}: {e}", flush=True)

    delta_acc_str = f"{(acc - baseline_acc):+.2f}%" if (acc is not None and baseline_acc is not None) else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    comp_rate_str = f"{comp_rate:.2f}x" if comp_rate is not None else "1.00x"
    vram_str = f"{peak_vram:.1f} MB" if peak_vram is not None else "N/A"

    final_stats.append({
        "Model": m["name"],
        "Exit Code": res.returncode,
        "Baseline Acc": f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A",
        "XAttention Acc": f"{acc:.2f}%" if acc is not None else "N/A",
        "Delta Acc": delta_acc_str,
        "F1": f1_str,
        "Status": status,
        "Compressed Layers": num_compressed_layers,
        "Comp Rate": comp_rate_str,
        "Peak VRAM": vram_str,
        "Time (s)": f"{elapsed:.1f}",
    })
    print(f"\n>>> FINISHED MODEL: {m['name']} in {elapsed:.1f}s (Exit: {res.returncode}, Status: {status})", flush=True)

print("\n" + "=" * 80, flush=True)
print("4-MODEL BENCHMARK RESULTS SUMMARY (XAttention on GPU)", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(final_stats)
print(res_df.to_string(index=False), flush=True)
print("=" * 80, flush=True)

out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "unified_xattention_4models_summary.csv"
out_json = out_dir / "unified_xattention_4models_summary.json"
res_df.to_csv(out_csv, index=False)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(final_stats, f, indent=2)
print(f"\nSaved consolidated summaries to:\n- {out_csv}\n- {out_json}", flush=True)

# Generate and save comparative plots in results directory
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _mod_slug = res_df.columns[2].replace(" Acc", "").lower() if len(res_df.columns) > 2 else "module"
    _plots_dir = out_dir / f"unified_{_mod_slug}"
    _plots_dir.mkdir(parents=True, exist_ok=True)

    _p_models = [str(s.get("Model", "")) for s in final_stats]
    _b_accs = [float(str(s.get("Baseline Acc", "0")).replace("%", "")) if str(s.get("Baseline Acc", "N/A")) != "N/A" else 0.0 for s in final_stats]
    _m_acc_col = [k for k in final_stats[0].keys() if "Acc" in k and k not in ("Baseline Acc", "Delta Acc")][0] if final_stats else "Acc"
    _m_accs = [float(str(s.get(_m_acc_col, "0")).replace("%", "")) if str(s.get(_m_acc_col, "N/A")) != "N/A" else 0.0 for s in final_stats]
    _c_rates = [float(str(s.get("Comp Rate", "1.0")).replace("x", "")) if str(s.get("Comp Rate", "N/A")) != "N/A" else 1.0 for s in final_stats]

    # Chart 1: Accuracy Comparison
    _x = np.arange(len(_p_models))
    _w = 0.35
    _fig, _ax1 = plt.subplots(figsize=(9, 5))
    _r1 = _ax1.bar(_x - _w/2, _b_accs, _w, label='Baseline Acc', color='#4C72B0')
    _r2 = _ax1.bar(_x + _w/2, _m_accs, _w, label=f'{_mod_slug.upper()} Acc', color='#55A868')
    _ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    _ax1.set_title(f'Unified 4-Model Quality Evaluation ({_mod_slug.upper()})', fontweight='bold')
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_p_models, fontweight='bold')
    _ax1.set_ylim(0, 105)
    _ax1.legend(loc='upper right')
    _ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for _rect in _r1:
        _h = _rect.get_height()
        if _h > 0: _ax1.annotate(f'{_h:.1f}%', xy=(_rect.get_x() + _rect.get_width()/2, _h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for _rect in _r2:
        _h = _rect.get_height()
        if _h > 0: _ax1.annotate(f'{_h:.1f}%', xy=(_rect.get_x() + _rect.get_width()/2, _h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(_plots_dir / f"{_mod_slug}_4models_accuracy_comparison.png", dpi=300)
    plt.close()

    # Chart 2: Pareto Frontier
    _fig, _ax2 = plt.subplots(figsize=(8, 5))
    for _i, _m_name in enumerate(_p_models):
        if _m_accs[_i] > 0:
            _ax2.scatter(_c_rates[_i], _m_accs[_i], s=160, label=_m_name, zorder=5)
            _ax2.annotate(f"{_m_name} ({_c_rates[_i]:.2f}x, {_m_accs[_i]:.1f}%)", (_c_rates[_i], _m_accs[_i]), xytext=(7, -4), textcoords='offset points', fontsize=9)
    _ax2.set_xlabel('Compression Rate (x)', fontweight='bold')
    _ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    _ax2.set_title(f'Pareto Frontier: Accuracy vs Compression ({_mod_slug.upper()})', fontweight='bold')
    _ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(_plots_dir / f"{_mod_slug}_pareto_accuracy_vs_compression.png", dpi=300)
    plt.close()
    print(f"Generated and saved comparative visualization charts to {_plots_dir}", flush=True)
except Exception as _e_p:
    print(f"[WARN] Could not generate comparative plots: {_e_p}", flush=True)


# Quality gate & export to HuggingFace
print("\n" + "=" * 80, flush=True)
print("STAGE 2: QUALITY-GATED EXPORT TO HUGGING FACE & GITHUB (XAttention)", flush=True)
print("=" * 80, flush=True)

from huggingface_hub import HfApi

hf_token = os.environ.get("HF_TOKEN")
api = HfApi(token=hf_token)
hf_org = "Hooshaai"
pass_count = 0
uploaded_repos = []

for m in models:
    name = m["name"]
    slug = m.get("slug", name)
    out_dir = Path(m["out_dir"])
    weights_dir = out_dir / "checkpoints" / "xattention"
    hf_export_dir = weights_dir / "hf_export"

    csv_path = Path(m["summary_csv"])
    row = {}
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            row_match = df[df["module_name"] == "xattention"]
            if not row_match.empty:
                row = row_match.iloc[-1].to_dict()
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

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-xattention"
    try:
        print(f"Uploading to Hugging Face: {repo_id}...", flush=True)
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(hf_export_dir),
            repo_id=repo_id,
            commit_message=f"Upload hardened XAttention model weights ({name}) - Acc: {acc:.2f}%",
        )
        print(f"Successfully uploaded {name} to {repo_id}", flush=True)
        uploaded_repos.append(repo_id)
    except Exception as e:
        print(f"Upload failed for {repo_id}: {e}", flush=True)

print(f"\nExport Complete: {pass_count}/{len(models)} passed quality gate.", flush=True)
