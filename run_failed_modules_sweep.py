#!/usr/bin/env python3
"""
Dedicated sweep orchestrator to re-benchmark all repaired modules across DistilBERT, RoBERTa, and GPT-2.
Automatically deletes generated model weights (.pt/.bin) after evaluation to keep disk clean.
"""

import os
import sys
import glob
import subprocess
import time
from pathlib import Path

PYTHON_EXEC = r"C:\Users\Hadi Schumacher\anaconda3\python.exe"
REPO_DIR = Path(r"D:\svd_project")
RESULTS_DIR = REPO_DIR / "results"

MODULES_TO_BENCHMARK = [
    "routing",
    "rala",
    "wgqa",
    "spectron",
    "spectral_mamba",
    "vspec",
    "simple_ssm",
    "tpa",
    "token_compressed",
    "qcqa",
    "upi_stuffed_mamba",
    "tt_linear",
    "tlinformer",
    "proxsparse",
    "slicegpt_pca",
    "padre",
    "spectraformer",
    "salr",
    "soka",
    "subspace_projection",
    "weighted_ensemble",
    "svd_compressed",
]

MODELS = ["distilbert", "roberta", "gpt2"]

def clean_checkpoints():
    removed = 0
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "baseline" in root.lower():
            continue
        for f in files:
            if f.endswith((".pt", ".bin", ".safetensors")):
                p = os.path.join(root, f)
                try:
                    os.remove(p)
                    removed += 1
                except Exception:
                    pass
    if removed > 0:
        print(f"[CLEANUP] Deleted {removed} intermediate model weight files.")

def run():
    print(f"=== Starting Re-evaluation of Repaired Modules ({len(MODULES_TO_BENCHMARK)} modules) ===")
    
    for mod in MODULES_TO_BENCHMARK:
        # Check if there is a runner script run_all_4_<mod>.py
        runner_candidates = list(REPO_DIR.glob(f"run_all_4_{mod}*.py"))
        
        if runner_candidates:
            runner = runner_candidates[0]
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> Running {runner.name} across all 3 models...")
            cmd = [
                PYTHON_EXEC,
                "-u",
                str(runner),
                "--models",
                ",".join(MODELS),
                "--recovery_steps",
                "50",
            ]
            try:
                subprocess.run(cmd, cwd=str(REPO_DIR), check=False)
            except Exception as e:
                print(f"Error running {runner.name}: {e}")
            clean_checkpoints()
        else:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> Module {mod} has no standalone runner, running quality sweep...")
            for m in MODELS:
                model_path = f"models\\{m}_sst2" if m == "distilbert" else (r"models\roberta-base" if m == "roberta" else r"models\gpt2")
                family = "bert" if m == "distilbert" else ("roberta" if m == "roberta" else "gpt")
                out_dir = str(RESULTS_DIR / f"unified_{mod}" / m)
                
                cmd = [
                    PYTHON_EXEC,
                    "-u",
                    str(REPO_DIR / "Codes" / "framework" / "run_quality_sweep.py"),
                    "--model", model_path,
                    "--model_family", family,
                    "--dataset", "sst2",
                    "--dataset_dir", str(REPO_DIR / "Codes" / "framework" / "data" / "glue_data" / "sst2"),
                    "--output_dir", out_dir,
                    "--modules", mod,
                    "--batch_size", "8",
                    "--recovery_steps", "50",
                    "--learning_rate", "2e-4",
                    "--lora_rank", "16",
                    "--precision", "fp16",
                    "--fresh",
                    "--defer_export"
                ]
                try:
                    subprocess.run(cmd, cwd=str(REPO_DIR), check=False)
                except Exception as e:
                    print(f"Error sweeping {mod} on {m}: {e}")
                clean_checkpoints()

if __name__ == "__main__":
    run()
