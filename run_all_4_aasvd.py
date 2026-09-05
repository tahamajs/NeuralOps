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
            "--rank", "288",
            "--aasvd_max_error", "0.60",
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

if len(sys.argv) > 1:
    selected = set(sys.argv[1:])
    models = [m for m in models if m["name"] in selected]

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED AASVD BENCHMARK ON GPU SERVER", flush=True)
print("Config: modules=full,aasvd, dataset=sst2, recovery_steps=200, lr=2e-4, rank=288/384, max_error=0.50, attention_only=True, batch_size=8, precision=fp16", flush=True)
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

