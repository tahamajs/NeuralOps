#!/usr/bin/env python3
"""
Sequential Rank Sweep Runner.
Executes Category 5A SVD compressor rank sweeps sequentially (one at a time)
across DistilBERT (SST-2), RoBERTa (SST-2), and GPT-2 (WikiText-2).
Avoids GPU memory contention and concurrency crashes on 8GB GPUs.
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
MASTER_LOG = LOGS_DIR / "sequential_rank_sweep.log"

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Read modules from cat5a_modules.txt
cat5a_file = BASE_DIR / "audit" / "cat5a_modules.txt"
if cat5a_file.exists():
    with open(cat5a_file, "r") as f:
        CAT5A_MODULES = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    CAT5A_MODULES = [
        "a3_compressor",
        "adasvd_compressor",
        "svd_llm_compressor",
        "saes_svd_compressor",
        "slicegpt",
        "swift_svd_act",
        "tt_compressor",
    ]

GPT2_MODULES = [
    "a3_compressor",
    "svd_llm_compressor",
    "saes_svd_compressor",
    "slicegpt",
    "swift_svd_act",
    "tt_compressor",
]

BERT_RANKS = [16, 32, 64, 128, 256, 384]
GPT2_RANKS = [16, 32, 64, 128, 256]

TASKS = []

# 1. DistilBERT SST-2
for m in CAT5A_MODULES:
    for r in BERT_RANKS:
        out_dir = BASE_DIR / "results" / "rank_sweep_sst2" / m / f"rank_{r}"
        TASKS.append({
            "name": f"DistilBERT SST-2 | {m} | rank={r}",
            "module": m,
            "model": "models/distilbert_sst2",
            "model_family": "bert",
            "dataset": "sst2",
            "rank": r,
            "lr": "2e-4",
            "recovery_steps": 50,
            "out_dir": str(out_dir),
            "summary_file": out_dir / "summary.csv",
        })

# 2. RoBERTa SST-2
for m in CAT5A_MODULES:
    for r in BERT_RANKS:
        out_dir = BASE_DIR / "results" / "rank_sweep_roberta" / m / f"rank_{r}"
        TASKS.append({
            "name": f"RoBERTa SST-2 | {m} | rank={r}",
            "module": m,
            "model": "models/roberta_sst2",
            "model_family": "roberta",
            "dataset": "sst2",
            "rank": r,
            "lr": "5e-5",
            "recovery_steps": 50,
            "out_dir": str(out_dir),
            "summary_file": out_dir / "summary.csv",
        })

# 3. GPT-2 WikiText-2
for m in GPT2_MODULES:
    for r in GPT2_RANKS:
        out_dir = BASE_DIR / "results" / "rank_sweep_gpt2" / m / f"rank_{r}"
        TASKS.append({
            "name": f"GPT-2 WikiText-2 | {m} | rank={r}",
            "module": m,
            "model": "models/gpt2",
            "model_family": "gpt2",
            "dataset": "wikitext2",
            "rank": r,
            "lr": "5e-5",
            "recovery_steps": 50,
            "out_dir": str(out_dir),
            "summary_file": out_dir / "summary.csv",
        })

def run_task(task):
    summary_path = Path(task["summary_file"])
    if summary_path.exists() and summary_path.stat().st_size > 50:
        # Check if it was a real pass (not a crash dummy)
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "CRASH_NO_OUTPUT" not in content and ("PASS" in content or "FAIL" in content):
                log(f"[SKIP] {task['name']} - already completed ({summary_path})")
                return True
        except Exception:
            pass

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
        "--recovery_steps", str(task["recovery_steps"]),
        "--learning_rate", task["lr"],
        "--seed", "42",
        "--output_dir", task["out_dir"],
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"Codes;{env.get('PYTHONPATH', '')}" if os.name == "nt" else f"Codes:{env.get('PYTHONPATH', '')}"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"[PASS] {task['name']} completed in {elapsed:.1f}s")
            return True
        else:
            log(f"[FAIL] {task['name']} failed with code {proc.returncode} in {elapsed:.1f}s")
            # Log tail of error
            lines = proc.stdout.strip().split("\n")
            tail = "\n".join(lines[-10:])
            log(f"Error tail:\n{tail}")
            return False
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] {task['name']} timed out after 3600s")
        return False
    except Exception as e:
        log(f"[EXCEPTION] {task['name']}: {e}")
        return False

def main():
    log(f"Starting Sequential Rank Sweep Runner ({len(TASKS)} tasks queued)...")
    completed = 0
    skipped = 0
    failed = 0

    for i, task in enumerate(TASKS):
        log(f"Progress: [{i+1}/{len(TASKS)}] (Done: {completed}, Skipped: {skipped}, Failed: {failed})")
        # Ensure GPU has settled
        time.sleep(2)
        summary_path = Path(task["summary_file"])
        if summary_path.exists() and summary_path.stat().st_size > 50:
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    c = f.read()
                if "CRASH_NO_OUTPUT" not in c and ("PASS" in c or "FAIL" in c):
                    skipped += 1
                    continue
            except Exception:
                pass

        success = run_task(task)
        if success:
            completed += 1
        else:
            failed += 1

    log(f"============================================================")
    log(f"Sequential Rank Sweep Runner FINISHED!")
    log(f"Total: {len(TASKS)} | Completed: {completed} | Skipped: {skipped} | Failed: {failed}")
    log(f"============================================================")

if __name__ == "__main__":
    main()
