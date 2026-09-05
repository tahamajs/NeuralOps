import subprocess
import os
import sys
import time
import pandas as pd
from pathlib import Path

os.chdir(r"D:\svd_project")

models = [
    {
        "name": "distilbert",
        "cmd": [
            sys.executable, "-u", "Codes\\framework\\run_quality_sweep.py",
            "--model", "models\\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", "Codes\\framework\\data\\glue_data\\sst2",
            "--output_dir", "results\\unified_aasvd\\distilbert",
            "--modules", "aasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "32",
            "--timeout", "600",
            "--precision", "fp16",
            "--fresh",
        ],
        "summary_csv": r"results\unified_aasvd\distilbert\summary.csv",
    },
    {
        "name": "roberta",
        "cmd": [
            sys.executable, "-u", "Codes\\framework\\run_quality_sweep.py",
            "--model", "models\\roberta-base",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", "Codes\\framework\\data\\glue_data\\sst2",
            "--output_dir", "results\\unified_aasvd\\roberta",
            "--modules", "aasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "32",
            "--timeout", "600",
            "--precision", "fp16",
            "--parent_checkpoint", "results\\roberta_sst2_lora_baseline\\finetuned_model.pt",
            "--fresh",
        ],
        "summary_csv": r"results\unified_aasvd\roberta\summary.csv",
    },
    {
        "name": "gpt2",
        "cmd": [
            sys.executable, "-u", "Codes\\framework\\run_quality_sweep.py",
            "--model", "models\\gpt2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", "Codes\\framework\\data\\glue_data\\sst2",
            "--output_dir", "results\\unified_aasvd\\gpt2",
            "--modules", "aasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "32",
            "--timeout", "600",
            "--precision", "fp16",
            "--parent_checkpoint", "results\\gpt2_sst2_lora_baseline\\finetuned_model.pt",
            "--fresh",
        ],
        "summary_csv": r"results\unified_aasvd\gpt2\summary.csv",
    },
    {
        "name": "qwen3_5",
        "cmd": [
            sys.executable, "-u", "Codes\\framework\\run_quality_sweep.py",
            "--model", "models\\Qwen3.5-0.8B",
            "--model_family", "qwen3_5",
            "--dataset", "sst2",
            "--dataset_dir", "Codes\\framework\\data\\glue_data\\sst2",
            "--output_dir", "results\\unified_aasvd\\qwen3_5",
            "--modules", "aasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--lora_rank", "32",
            "--timeout", "600",
            "--precision", "fp16",
            "--fresh",
        ],
        "summary_csv": r"results\unified_aasvd\qwen3_5\summary.csv",
    },
]

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED AASVD BENCHMARK ON GPU SERVER", flush=True)
print("Config: modules=aasvd, dataset=sst2, recovery_steps=50, lr=2e-4, lora_rank=32, batch_size=8, precision=fp16", flush=True)
print("=" * 80, flush=True)

final_stats = []

for m in models:
    print(f"\n{'='*30} RUNNING MODEL: {m['name']} {'='*30}", flush=True)
    t0 = time.time()
    res = subprocess.run(" ".join(m["cmd"]), shell=True)
    elapsed = time.time() - t0
    
    acc = None
    status = "UNKNOWN"
    csv_path = Path(m["summary_csv"])
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                acc = df.iloc[-1].get("eval_accuracy", df.iloc[-1].get("accuracy", None))
                status = df.iloc[-1].get("status", "DONE")
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}")
            
    final_stats.append({
        "Model": m["name"],
        "Exit Code": res.returncode,
        "Accuracy": f"{acc*100:.2f}%" if acc is not None else "N/A",
        "Status": status,
        "Time (s)": f"{elapsed:.1f}",
    })
    print(f">>> FINISHED MODEL: {m['name']} in {elapsed:.1f}s (Exit: {res.returncode})", flush=True)

print("\n" + "=" * 80, flush=True)
print("4-MODEL BENCHMARK RESULTS SUMMARY (AASVD on GPU)", flush=True)
print("=" * 80, flush=True)
for s in final_stats:
    print(f"  Model: {s['Model']:<12} | Status: {s['Status']:<10} | Accuracy: {s['Accuracy']:<8} | Time: {s['Time (s)']}s")
print("=" * 80, flush=True)
