#!/usr/bin/env python3
"""
Multi-Dataset Runner for A3 Compressor across 3 Models (DistilBERT, RoBERTa, GPT-2)
on MNLI and AG News.
"""

import os
import sys
import time
import json
import subprocess
import pandas as pd
from pathlib import Path

PYTHON_EXE = r"sys.executable"
BASE_DIR = r"PROJECT_ROOT"

tasks = [
    # --- MNLI Benchmark (3 classes) ---
    {
        "model_name": "DistilBERT",
        "dataset": "mnli",
        "output_dir": r"results\a3_tests\datasets\mnli\distilbert",
        "cmd": (
            f'"{PYTHON_EXE}" -u Codes\\framework\\run_quality_sweep.py '
            r"--model models\distilbert_sst2 --model_family bert --dataset mnli "
            r"--dataset_dir Codes\framework\data\glue_data\mnli "
            r"--output_dir results\a3_tests\datasets\mnli\distilbert "
            r"--modules a3 --batch_size 8 --recovery_steps 50 --lora_rank 16 --precision fp16 --fresh"
        ),
    },
    {
        "model_name": "RoBERTa",
        "dataset": "mnli",
        "output_dir": r"results\a3_tests\datasets\mnli\roberta",
        "cmd": (
            f'"{PYTHON_EXE}" -u Codes\\framework\\run_quality_sweep.py '
            r"--model models\roberta-base --model_family roberta --dataset mnli "
            r"--dataset_dir Codes\framework\data\glue_data\mnli "
            r"--output_dir results\a3_tests\datasets\mnli\roberta "
            r"--modules a3 --batch_size 4 --recovery_steps 50 --lora_rank 8 --precision fp16 --fresh"
        ),
    },
    {
        "model_name": "GPT-2",
        "dataset": "mnli",
        "output_dir": r"results\a3_tests\datasets\mnli\gpt2",
        "cmd": (
            f'"{PYTHON_EXE}" -u Codes\\framework\\run_quality_sweep.py '
            r"--model models\gpt2 --model_family gpt --dataset mnli "
            r"--dataset_dir Codes\framework\data\glue_data\mnli "
            r"--output_dir results\a3_tests\datasets\mnli\gpt2 "
            r"--modules a3 --batch_size 4 --recovery_steps 50 --lora_rank 8 --precision fp16 --fresh"
        ),
    },
    # --- AG News Benchmark (4 classes) ---
    {
        "model_name": "DistilBERT",
        "dataset": "ag_news",
        "output_dir": r"results\a3_tests\datasets\agnews\distilbert",
        "cmd": (
            f'"{PYTHON_EXE}" -u Codes\\framework\\run_quality_sweep.py '
            r"--model models\distilbert_sst2 --model_family bert --dataset ag_news "
            r"--dataset_dir Codes\framework\data\glue_data\ag_news "
            r"--output_dir results\a3_tests\datasets\agnews\distilbert "
            r"--modules a3 --batch_size 8 --recovery_steps 50 --lora_rank 16 --precision fp16 --fresh"
        ),
    },
    {
        "model_name": "RoBERTa",
        "dataset": "ag_news",
        "output_dir": r"results\a3_tests\datasets\agnews\roberta",
        "cmd": (
            f'"{PYTHON_EXE}" -u Codes\\framework\\run_quality_sweep.py '
            r"--model models\roberta-base --model_family roberta --dataset ag_news "
            r"--dataset_dir Codes\framework\data\glue_data\ag_news "
            r"--output_dir results\a3_tests\datasets\agnews\roberta "
            r"--modules a3 --batch_size 4 --recovery_steps 50 --lora_rank 8 --precision fp16 --fresh"
        ),
    },
    {
        "model_name": "GPT-2",
        "dataset": "ag_news",
        "output_dir": r"results\a3_tests\datasets\agnews\gpt2",
        "cmd": (
            f'"{PYTHON_EXE}" -u Codes\\framework\\run_quality_sweep.py '
            r"--model models\gpt2 --model_family gpt --dataset ag_news "
            r"--dataset_dir Codes\framework\data\glue_data\ag_news "
            r"--output_dir results\a3_tests\datasets\agnews\gpt2 "
            r"--modules a3 --batch_size 4 --recovery_steps 50 --lora_rank 8 --precision fp16 --fresh"
        ),
    },
]

def main():
    os.chdir(BASE_DIR)
    results = []
    
    print("=" * 80, flush=True)
    print("STARTING A3 MULTI-DATASET SWEEP (MNLI & AG NEWS across 3 Models)", flush=True)
    print("=" * 80, flush=True)
    
    for t in tasks:
        print(f"\n>>> Running {t['model_name']} on {t['dataset'].upper()}...", flush=True)
        t0 = time.time()
        res = subprocess.run(t["cmd"], shell=True)
        elapsed = time.time() - t0
        
        summary_csv = Path(BASE_DIR) / t["output_dir"] / "summary.csv"
        acc = 0.0
        status = "FAIL"
        f1 = 0.0
        if summary_csv.exists():
            try:
                df = pd.read_csv(summary_csv)
                row = df[df["module_name"] == "a3"]
                if not row.empty:
                    acc = float(row["accuracy_pct"].values[0])
                    status = str(row["status"].values[0])
                    f1 = float(row["f1"].values[0]) if "f1" in row else 0.0
            except Exception as e:
                print(f"Error parsing {summary_csv}: {e}")
                
        record = {
            "Model": t["model_name"],
            "Dataset": t["dataset"],
            "Status": status,
            "Accuracy (%)": acc,
            "F1": f1,
            "Elapsed (s)": round(elapsed, 1),
        }
        results.append(record)
        print(f">>> [FINISHED] {t['model_name']} on {t['dataset']}: {status} (Acc: {acc}%, F1: {f1}) in {elapsed:.1f}s", flush=True)
        
    out_csv = Path(BASE_DIR) / "results" / "a3_tests" / "multidataset_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80, flush=True)
    print("MULTI-DATASET SWEEP COMPLETE", flush=True)
    print(res_df.to_string(index=False), flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
