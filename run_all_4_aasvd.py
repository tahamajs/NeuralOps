import subprocess
import os
import sys
import time
import json
import pandas as pd
from pathlib import Path

# Switch to project directory if on Windows GPU server
if os.path.exists(r"D:\svd_project"):
    os.chdir(r"D:\svd_project")

models = [
    {
        "name": "distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--modules", "aasvd",
            "--models", "distilbert",
            "--datasets", "sst2",
            "--batch_size", "8",
            "--precision", "fp16",
            "--max_steps", "50",
            "--output_dir", r"results\unified_aasvd\distilbert",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\distilbert\summary.csv",
    },
    {
        "name": "roberta",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--modules", "aasvd",
            "--models", "roberta",
            "--datasets", "sst2",
            "--batch_size", "8",
            "--precision", "fp16",
            "--max_steps", "50",
            "--output_dir", r"results\unified_aasvd\roberta",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\roberta\summary.csv",
    },
    {
        "name": "gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--modules", "aasvd",
            "--models", "gpt2",
            "--datasets", "sst2",
            "--batch_size", "8",
            "--precision", "fp16",
            "--max_steps", "50",
            "--output_dir", r"results\unified_aasvd\gpt2",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\gpt2\summary.csv",
    },
    {
        "name": "qwen3_5",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--modules", "aasvd",
            "--models", "qwen3_5",
            "--datasets", "sst2",
            "--batch_size", "8",
            "--precision", "fp16",
            "--max_steps", "50",
            "--output_dir", r"results\unified_aasvd\qwen3_5",
            "--timeout", "1200",
        ],
        "summary_csv": r"results\unified_aasvd\qwen3_5\summary.csv",
    },
]

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED AASVD BENCHMARK ON GPU SERVER", flush=True)
print("Config: modules=aasvd, dataset=sst2, max_steps=50, batch_size=8, precision=fp16", flush=True)
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
    csv_path = Path(m["summary_csv"])
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                row = df[df["module_name"] == "aasvd"]
                if not row.empty:
                    acc = float(row["mean_accuracy"].values[-1])
                    status = str(row["status"].values[-1])
                row_base = df[df["module_name"] == "full"]
                if not row_base.empty:
                    baseline_acc = float(row_base["mean_accuracy"].values[-1])
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}", flush=True)
            
    final_stats.append({
        "Model": m["name"],
        "Exit Code": res.returncode,
        "Baseline Acc": f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A",
        "AASVD Acc": f"{acc:.2f}%" if acc is not None else "N/A",
        "Status": status,
        "Time (s)": f"{elapsed:.1f}",
    })
    print(f"\n>>> FINISHED MODEL: {m['name']} in {elapsed:.1f}s (Exit: {res.returncode}, Status: {status})", flush=True)

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

