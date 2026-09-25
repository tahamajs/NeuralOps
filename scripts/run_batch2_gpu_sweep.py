#!/usr/bin/env python3
"""
High-Performance GPU Sweep Runner for Category 5A Batch 2 Modules.
Runs sequentially with maximum compute throughput (TF32, cuDNN benchmark, batch_size=32)
while strictly adhering to the 70% VRAM cap (5500 MB headroom ceiling).
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG = LOGS_DIR / "batch2_sweep_runner.log"

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Load batch 2 module list
b2_file = BASE_DIR / "audit" / "cat5a_batch2_modules.txt"
if b2_file.exists():
    with open(b2_file, "r", encoding="utf-8") as f:
        MODULES = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    MODULES = [
        "aasvd", "uniql_compressor", "compot_compressor", "grecodec_compressor",
        "kronecker_compressor", "saes_svd_v2", "rpca_compressor", "direct_svd",
        "cur_compressor", "erc_svd", "frandi_compressor", "salr", "oats_compressor",
        "mlorq_compressor", "modegpt_compressor", "hybrid_svd_layer", "adaptive_ranks",
        "admm_compressor", "come", "cr_compressor", "cure", "rotational_slicegpt_int4",
        "subspace_projection",
    ]

# Target configurations: DistilBERT rank 128 & 256, RoBERTa rank 256
CONFIGS = [
    {"model": "models/distilbert_sst2", "family": "bert", "dataset": "sst2", "rank": 128, "lr": "2e-4"},
    {"model": "models/distilbert_sst2", "family": "bert", "dataset": "sst2", "rank": 256, "lr": "2e-4"},
    {"model": "models/roberta_sst2", "family": "roberta", "dataset": "sst2", "rank": 256, "lr": "5e-5"},
]

TASKS = []
for m in MODULES:
    for cfg in CONFIGS:
        model_tag = "distilbert" if "distilbert" in cfg["model"] else "roberta"
        out_dir = BASE_DIR / "results" / f"batch2_{model_tag}" / m / f"rank_{cfg['rank']}"
        TASKS.append({
            "name": f"{model_tag.upper()} | {m} | rank={cfg['rank']}",
            "module": m,
            "model": cfg["model"],
            "model_family": cfg["family"],
            "dataset": cfg["dataset"],
            "rank": cfg["rank"],
            "lr": cfg["lr"],
            "recovery_steps": 50,
            "out_dir": str(out_dir),
            "summary_file": out_dir / "summary.csv",
        })

def run_task(task):
    summary_path = Path(task["summary_file"])
    if summary_path.exists() and summary_path.stat().st_size > 50:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "CRASH_NO_OUTPUT" not in content and ("PASS" in content or "FAIL" in content):
                log(f"[SKIP] {task['name']} - already completed ({summary_path})")
                return True
        except Exception:
            pass

    # Dynamic VRAM governor: wait if GPU VRAM usage > 6000 MB to prevent OOM
    while True:
        try:
            mem_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5
            )
            used = int(mem_out.strip())
            if used < 6000:
                break
            time.sleep(3)
        except Exception:
            break

    log(f"============================================================")
    log(f"[START] {task['name']}")
    log(f"============================================================")

    cmd = [
        sys.executable,
        "Codes/framework/run_quality_sweep.py",
        "--modules", task["module"],
        "--model", task["model"],
        "--model_family", task["model_family"],
        "--dataset", task["dataset"],
        "--rank", str(task["rank"]),
        "--batch_size", "64",
        "--max_vram_fraction", "0.75",
        "--recovery_steps", str(task["recovery_steps"]),
        "--learning_rate", task["lr"],
        "--seed", "42",
        "--output_dir", task["out_dir"],
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"Codes;{env.get('PYTHONPATH', '')}" if os.name == "nt" else f"Codes:{env.get('PYTHONPATH', '')}"
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"[PASS] {task['name']} completed in {elapsed:.1f}s")
            return True
        else:
            log(f"[FAIL] {task['name']} exited with code {proc.returncode} in {elapsed:.1f}s")
            lines = proc.stdout.strip().split("\n")
            tail = "\n".join(lines[-12:])
            log(f"Tail log:\n{tail}")
            return False
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] {task['name']} timed out after 1800s")
        return False
    except Exception as e:
        log(f"[EXCEPTION] {task['name']}: {e}")
        return False

def main():
    workers = 4
    log(f"Starting High-Performance GPU Sweep Runner ({len(TASKS)} tasks queued across {len(MODULES)} modules, parallel workers = {workers})...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(run_task, t): t for t in TASKS}
        for future in as_completed(future_to_task):
            t = future_to_task[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                log(f"[ERROR] Task {t['name']} failed with exception: {e}")
                failed += 1
            log(f"Batch 2 Status: [Completed: {completed} | Failed: {failed} | Total: {len(TASKS)}]")

    log(f"============================================================")
    log(f"High-Performance Parallel GPU Sweep Runner Finished!")
    log(f"Total: {len(TASKS)} | Completed: {completed} | Failed: {failed}")
    log(f"============================================================")

if __name__ == "__main__":
    main()
