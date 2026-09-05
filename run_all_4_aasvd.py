import subprocess
import os
import sys
import time
import json
import pandas as pd
from pathlib import Path

# Enforce completely offline dataset and model resolution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Switch to project directory if on Windows GPU server
if os.path.exists(r"D:\svd_project"):
    os.chdir(r"D:\svd_project")

models = [
    {
        "name": "distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "200",
            "--learning_rate", "2e-4",
            "--rank", "288",
            "--aasvd_max_error", "0.50",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\distilbert\summary.csv",
        "stats_json": r"results\unified_aasvd\distilbert\aasvd_stats.json",
    },
    {
        "name": "roberta",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\roberta-base",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "200",
            "--learning_rate", "2e-4",
            "--rank", "288",
            "--aasvd_max_error", "0.50",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\roberta\summary.csv",
        "stats_json": r"results\unified_aasvd\roberta\aasvd_stats.json",
    },
    {
        "name": "gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\gpt2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "200",
            "--learning_rate", "2e-4",
            "--rank", "432",
            "--aasvd_max_error", "0.70",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\gpt2\summary.csv",
        "stats_json": r"results\unified_aasvd\gpt2\aasvd_stats.json",
    },
    {
        "name": "qwen3_5",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\Qwen3.5-0.8B",
            "--model_family", "qwen3_5",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\qwen3_5",
            "--parent_checkpoint", r"results\unified_asvd\qwen3_5\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "200",
            "--learning_rate", "2e-4",
            "--rank", "384",
            "--aasvd_max_error", "0.50",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--timeout", "1500",
        ],
        "summary_csv": r"results\unified_aasvd\qwen3_5\summary.csv",
        "stats_json": r"results\unified_aasvd\qwen3_5\aasvd_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model Hardened AASVD Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,aasvd)")
parser.add_argument("--recovery_steps", type=int, default=None, help="Number of recovery fine-tuning steps")
parser.add_argument("--rank", type=int, default=None, help="Compression rank override")
parser.add_argument("--max_error", type=float, default=None, help="AASVD max reconstruction error threshold")
parser.add_argument("--attention_only", type=str, default=None, help="Target attention layers only (true/false)")
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

def update_cmd_arg(cmd: list[str], flag: str, new_val: str):
    try:
        idx = cmd.index(flag)
        cmd[idx + 1] = str(new_val)
    except ValueError:
        cmd.extend([flag, str(new_val)])

for m in models:
    if args.modules is not None:
        update_cmd_arg(m["cmd"], "--modules", args.modules)
    if args.recovery_steps is not None:
        update_cmd_arg(m["cmd"], "--recovery_steps", str(args.recovery_steps))
    if args.rank is not None:
        update_cmd_arg(m["cmd"], "--rank", str(args.rank))
    if args.max_error is not None:
        update_cmd_arg(m["cmd"], "--aasvd_max_error", str(args.max_error))
    if args.attention_only is not None:
        val = str(args.attention_only).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--aasvd_target_attention_only", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--aasvd_target_attention_only", "false")

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED AASVD BENCHMARK ON GPU SERVER", flush=True)
print(f"Running models: {[m['name'] for m in models]}", flush=True)
print("=" * 80, flush=True)

final_stats = []

for m in models:
    print(f"\n{'='*30} RUNNING MODEL: {m['name']} {'='*30}", flush=True)
    t0 = time.time()
    
    # Run directly without shell=True to avoid path space issues with Python binary on Windows
    res = subprocess.run(m["cmd"])
    elapsed = time.time() - t0
    
    acc = None
    baseline_acc = None
    status = "UNKNOWN" if res.returncode == 0 else f"EXIT_{res.returncode}"
    num_compressed_layers = "N/A"
    
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
                    row = df[df["module_name"] == "aasvd"]
                    if not row.empty:
                        acc = float(row[col].values[-1])
                        status = str(row["status"].values[-1])
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}", flush=True)

    stats_path = Path(m.get("stats_json", ""))
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                num_compressed_layers = s_data.get("num_compressed_layers", "N/A")
        except Exception as e:
            print(f"Error reading stats JSON for {m['name']}: {e}", flush=True)
            
    final_stats.append({
        "Model": m["name"],
        "Exit Code": res.returncode,
        "Baseline Acc": f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A",
        "AASVD Acc": f"{acc:.2f}%" if acc is not None else "N/A",
        "Status": status,
        "Compressed Layers": num_compressed_layers,
        "Time (s)": f"{elapsed:.1f}",
    })
    print(f"\n>>> FINISHED MODEL: {m['name']} in {elapsed:.1f}s (Exit: {res.returncode}, Status: {status}, Compressed Layers: {num_compressed_layers})", flush=True)

print("\n" + "=" * 80, flush=True)
print("4-MODEL BENCHMARK RESULTS SUMMARY (AASVD on GPU)", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(final_stats)
print(res_df.to_string(index=False), flush=True)
print("=" * 80, flush=True)

# Save consolidated summary
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "unified_aasvd_4models_summary.csv"
out_json = out_dir / "unified_aasvd_4models_summary.json"
res_df.to_csv(out_csv, index=False)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(final_stats, f, indent=2)
print(f"\nSaved consolidated summaries to:\n- {out_csv}\n- {out_json}", flush=True)

