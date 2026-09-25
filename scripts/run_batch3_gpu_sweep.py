#!/usr/bin/env python3
"""
High-Performance GPU Sweep Runner for Category 5B Modules (Pruning, Quantization, KV Cache).
Runs with 4 parallel workers for full GPU compute utilization,
batch_size=64, and dynamic VRAM governor (< 6000 MB ceiling).
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
MASTER_LOG = LOGS_DIR / "batch3_sweep_runner.log"

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Load batch 3 module list
b3_file = BASE_DIR / "audit" / "cat5b_modules.txt"
if b3_file.exists():
    with open(b3_file, "r", encoding="utf-8") as f:
        MODULES = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    MODULES = [
        "admm_pruner", "armor_pruner", "ast_pruner", "astro_quantizer", "bonsai_pruner",
        "cachenotes_compressor", "coopq_quantizer", "delta_kv", "enlighten_compressor",
        "entropy_budget_allocation", "faq_quantizer", "flap", "flat_llm", "hashed_linear",
        "hashed_net_compressor", "huffman_coder", "hybridkv_compressor", "kmeans_quantize",
        "kvtc_compressor", "liftr", "lillama_compressor", "mamba_shedder", "proxsparse",
        "replaceme_compressor", "roast_hashed_linear", "slim_compressor", "soka_compressor",
        "sparsegpt", "zerokv_neo_compressor"
    ]

# Target configurations: DistilBERT and RoBERTa
CONFIGS = [
    {"model": "models/distilbert_sst2", "family": "bert", "dataset": "sst2", "lr": "2e-4"},
    {"model": "models/roberta_sst2", "family": "roberta", "dataset": "sst2", "lr": "5e-5"},
]

TASKS = []
for m in MODULES:
    for cfg in CONFIGS:
        model_tag = "distilbert" if "distilbert" in cfg["model"] else "roberta"
        out_dir = BASE_DIR / "results" / f"batch3_{model_tag}" / m
        TASKS.append({
            "name": f"{model_tag.upper()} | {m}",
            "module": m,
            "model": cfg["model"],
            "model_family": cfg["family"],
            "dataset": cfg["dataset"],
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

    # Dynamic VRAM governor: wait if GPU VRAM usage > 6000 MB
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
    log(f"Starting Category 5B GPU Sweep Runner ({len(TASKS)} tasks queued across {len(MODULES)} modules, parallel workers = {workers})...")
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
            log(f"Batch 3 Status: [Completed: {completed} | Failed: {failed} | Total: {len(TASKS)}]")

    log(f"============================================================")
    log(f"Category 5B GPU Sweep Runner Finished!")
    log(f"Total: {len(TASKS)} | Completed: {completed} | Failed: {failed}")
    log(f"============================================================")

if __name__ == "__main__":
    main()
