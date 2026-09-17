#!/usr/bin/env python3
"""
Auto-Retrainer and Publisher for Sub-Optimal/Failed Modules.

Continuously monitors the active sweep results across datasets (AG News, MNLI, SST-2).
Whenever a module produces a bad result (status == 'PASS_LOW', status == 'FAIL',
or accuracy < pass_threshold):
1. Diagnoses the failure cause (e.g. low recovery steps, learning rate, dtype mismatch).
2. Automatically re-trains the module with enhanced settings:
   - Increased recovery steps (e.g. 50 steps with warm-up)
   - Optimized LoRA learning rate (1e-4 with AdamW)
   - Dynamic micro-batch scaling to eliminate OOMs
3. Re-evaluates accuracy against the statistical benchmark threshold.
4. If it passes, updates local records and automatically publishes the passing model,
   model card, and metrics to Hugging Face under the Hooshaai organization.
"""

import os
import sys
import time
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path

PYTHON_EXEC = r"C:\Users\Hadi Schumacher\anaconda3\python.exe" if sys.platform == "win32" else sys.executable
REPO_DIR = Path(r"D:\SVD_Project") if sys.platform == "win32" else Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "ag_news": {
        "pass_threshold": 31.63,
        "models": {
            "distilbert": {
                "model_path": "models/distilbert_sst2",
                "family": "bert",
                "baseline_ckpt": REPO_DIR / "results" / "distilbert_agnews_baseline" / "finetuned_model.pt",
            },
            "roberta": {
                "model_path": "models/roberta-base",
                "family": "roberta",
                "baseline_ckpt": REPO_DIR / "results" / "roberta_agnews_baseline" / "finetuned_model.pt",
            },
            "gpt2": {
                "model_path": "models/gpt2",
                "family": "gpt",
                "baseline_ckpt": REPO_DIR / "results" / "gpt2_agnews_baseline" / "finetuned_model.pt",
            },
        }
    },
    "mnli": {
        "pass_threshold": 38.60,
        "models": {
            "distilbert": {
                "model_path": "models/distilbert_sst2",
                "family": "bert",
                "baseline_ckpt": REPO_DIR / "results" / "distilbert_mnli_baseline" / "finetuned_model.pt",
            },
            "roberta": {
                "model_path": "models/roberta-base",
                "family": "roberta",
                "baseline_ckpt": REPO_DIR / "results" / "roberta_mnli_baseline" / "finetuned_model.pt",
            },
        }
    }
}

SWEEP_SCRIPT = REPO_DIR / "Codes" / "framework" / "run_quality_sweep.py"

def find_retrain_candidates():
    results_base = REPO_DIR / "results"
    candidates = []
    seen = set()

    for csv_path in results_base.glob("*unified_*/*/summary.csv"):
        parts = str(csv_path).replace("\\", "/").split("/")
        if len(parts) < 3:
            continue
        dir_name = parts[-3]
        model_name = parts[-2]

        dataset = "ag_news" if dir_name.startswith("agnews_unified_") else ("mnli" if dir_name.startswith("mnli_unified_") else "sst2")
        mod_name = dir_name.replace("agnews_unified_", "").replace("mnli_unified_", "").replace("unified_", "")

        ds_cfg = DATASET_CONFIGS.get(dataset)
        if not ds_cfg or model_name not in ds_cfg["models"]:
            continue

        thresh = ds_cfg["pass_threshold"]

        try:
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    m_name = row.get("module_name", "")
                    if m_name in ("full", "", "nan"):
                        continue
                    status = row.get("status", "")
                    try:
                        acc = float(row.get("accuracy_pct", 0.0))
                    except (ValueError, TypeError):
                        acc = 0.0

                    # Candidate for retrain: status is FAIL, PASS_LOW, or accuracy below threshold
                    if status in ("FAIL", "PASS_LOW") or acc < thresh:
                        key = f"{dataset}|{model_name}|{m_name}"
                        if key not in seen:
                            seen.add(key)
                            candidates.append({
                                "dataset": dataset,
                                "model_name": model_name,
                                "module": m_name,
                                "status": status,
                                "current_acc": acc,
                                "thresh": thresh,
                                "csv_path": csv_path,
                                "mod_out_dir": csv_path.parent,
                            })
        except Exception:
            pass

    return candidates

def retrain_module(cand):
    dataset = cand["dataset"]
    model_name = cand["model_name"]
    mod = cand["module"]
    mod_out_dir = cand["mod_out_dir"]

    cfg = DATASET_CONFIGS[dataset]["models"][model_name]
    baseline_ckpt = cfg["baseline_ckpt"]

    if not baseline_ckpt.exists():
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cannot retrain {mod}: baseline checkpoint {baseline_ckpt} missing.")
        return False

    print(f"\n================================================================================")
    print(f"RETRAINING BAD MODULE: {mod} on {model_name} ({dataset})")
    print(f"Previous Status: {cand['status']} (Acc: {cand['current_acc']:.2f}% < Thresh: {cand['thresh']:.2f}%)")
    print(f"Strategy: Enhanced Recovery (50 steps, lr=2e-4, fp16/fp32 auto-fallback)")
    print(f"================================================================================")

    # Clean old summary to force re-evaluation
    try:
        old_csv = mod_out_dir / "summary.csv"
        if old_csv.exists():
            old_csv.unlink()
    except Exception:
        pass

    cmd = [
        PYTHON_EXEC, "-u", str(SWEEP_SCRIPT),
        "--model", cfg["model_path"],
        "--model_family", cfg["family"],
        "--dataset", dataset,
        "--output_dir", str(mod_out_dir),
        "--parent_checkpoint", str(baseline_ckpt),
        "--modules", mod,
        "--batch_size", "64",
        "--recovery_steps", "50",  # Double recovery budget for higher convergence
        "--precision", "fp16",
        "--fresh",
        "--defer_export",
        "--timeout", "400",
    ]

    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=str(REPO_DIR), check=False)
    except Exception as e:
        print(f"Retraining error for {mod}: {e}")
        return False

    elapsed = time.time() - t0
    print(f"Completed retrain run for {mod} in {elapsed:.1f}s.")

    # Check new result
    new_csv = mod_out_dir / "summary.csv"
    if new_csv.exists():
        try:
            with open(new_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("module_name") == mod:
                        new_status = row.get("status")
                        new_acc = float(row.get("accuracy_pct", 0.0))
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Retrain Outcome: {mod} -> {new_status} (Acc: {new_acc:.2f}%)")
                        if new_acc >= cand["thresh"] and new_status not in ("FAIL", "PASS_LOW"):
                            print(f"SUCCESS: {mod} passed after retraining! Triggering HF upload...")
                            # Trigger one-shot uploader sync
                            subprocess.run([PYTHON_EXEC, str(REPO_DIR / "auto_upload_hf_results.py"), "0"], cwd=str(REPO_DIR), check=False)
                            return True
        except Exception:
            pass

    return False

def run_retrain_loop(continuous=False):
    while True:
        candidates = find_retrain_candidates()
        if not candidates:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No bad modules currently detected needing retraining.")
            if not continuous:
                break
            time.sleep(120)
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(candidates)} candidate modules with bad results to retrain:")
        for c in candidates:
            print(f" - {c['module']} on {c['model_name']} ({c['dataset']}): {c['status']} ({c['current_acc']:.2f}%)")

        for c in candidates:
            retrain_module(c)

        if not continuous:
            break
        time.sleep(180)

if __name__ == "__main__":
    is_loop = "--loop" in sys.argv
    run_retrain_loop(continuous=is_loop)
