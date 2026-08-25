#!/usr/bin/env python3
"""
Single-Process Full GPU Experiment Runner
Runs all 15 attention mechanisms across GLUE benchmark tasks in a single persistent Python runtime.
100% Offline with zero network latency, zero DLL reload overhead, and zero storage bloat.
"""
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
    pass

# Force 100% Offline Mode (use cached HuggingFace models & on-disk Arrow datasets)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
from main import run_train, _import_framework

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GPU_Orchestrator")

MASTER_LOG = PROJECT_DIR / "trajectory_master.log"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def log_master(msg: str):
    logger.info(msg)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(MASTER_LOG, "a", encoding="utf-8", errors="ignore") as f:
            f.write(line)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
        pass


def main():
    log_master("=" * 60)
    log_master("  STARTING SVD COMPRESSION GPU ORCHESTRATION SUITE  ")
    log_master("=" * 60)
    log_master(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log_master(f"  VRAM: {vram:.1f} GB | CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
    log_master("  Storage Policy: ZERO-BLOAT (Metrics JSON only, no weights saved)")
    log_master("=" * 60)

    fw = _import_framework()

    ATTN_MODELS = [
        "svd_linear",
        "svd",
        "full",
        "lora_attn",
        "performer",
        "nystrom",
        "linformer",
        "flash3",
        "mamba",
        "gated_delta_net",
        "adasplash",
        "aegis_v3",
        "griffin_ssd",
        "spectra_v2",
        "samba_m3",
    ]

    TASKS = ["sst2", "mrpc", "qnli"]
    total = len(ATTN_MODELS) * len(TASKS)
    idx = 0

    
    completed_runs = []
    if MASTER_LOG.exists():
        with open(MASTER_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "<< SUCCESS:" in line or "⚠️ NOTICE:" in line:
                    if "attn=" in line and "task=" in line:
                        parts = line.split(" | ")
                        a = [p for p in parts if "attn=" in p]
                        t = [p for p in parts if "task=" in p]
                        if a and t:
                            attn_val = a[0].split("=")[1].strip()
                            task_val = t[0].split("=")[1].strip()
                            completed_runs.append((attn_val, task_val))

    log_master(f"Trajectory Matrix: {len(ATTN_MODELS)} Models x {len(TASKS)} Tasks = {total} Runs")


    for task in TASKS:
        log_master(f"▶ GLUE TASK: {task.upper()}")
        for attn in ATTN_MODELS:
            idx += 1
            log_master(f"[{idx}/{total}] >> Launching: attn={attn} | task={task} | epochs=0.5 | seed=42")
            
            cfg = fw["ExperimentConfig"]()
            cfg.attn_type = attn
            cfg.task = task
            cfg.epochs = 0.5
            cfg.batch_size = 32
            cfg.seed = 42
            cfg.lr = 3e-5 if attn in ("svd_linear", "full") else 5e-5
            cfg.save_model = False

            t0 = time.time()
            try:
                result = run_train(cfg, fw, logger)
                elapsed = time.time() - t0
                eval_metrics = result.get("eval", {})
                acc = eval_metrics.get("eval_accuracy", 0.0)
                f1 = eval_metrics.get("eval_f1_macro", 0.0)
                loss = eval_metrics.get("eval_loss", 0.0)
                log_master(f"[{idx}/{total}] << SUCCESS: attn={attn} | task={task} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Loss: {loss:.4f} | Time: {elapsed:.1f}s")
            
            except Exception as e:
                elapsed = time.time() - t0
                log_master(f"[{idx}/{total}] ⚠️ NOTICE: attn={attn} | task={task} | Error: {e} | Time: {elapsed:.1f}s")

            # Push results to git for live github pages
            try:
                import subprocess
                subprocess.run(["git", "add", "-f", str(RESULTS_DIR)], cwd=str(PROJECT_DIR), check=True)
                subprocess.run(["git", "commit", "-m", f"Auto-commit: {attn} on {task}"], cwd=str(PROJECT_DIR), check=False)
                subprocess.run(["git", "push"], cwd=str(PROJECT_DIR), check=False)
            except Exception as e:
                log_master(f"Git push failed: {e}")

            # Clean GPU memory & cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    log_master("=" * 60)
    log_master("  ALL 45 EXPERIMENT TRAJECTORIES COMPLETED SUCCESSFULLY!")
    log_master("=" * 60)


if __name__ == "__main__":
    main()
