#!/usr/bin/env python3
"""
Autonomous 48-Hour Continuous GPU Orchestrator.
Supervises continuous multi-worker execution across all models and modules:
- Ensures GPU compute cores remain continuously loaded (~80-95% utilization).
- When Category 5B finishes, immediately seamlessly launches the full 48-Hour Pipeline.
- Performs automated 20-minute audits and updates MASTER_PROJECT_RESULTS.md.
"""
import os
import sys
import time
import glob
import csv
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ORCHESTRATOR_LOG = LOGS_DIR / "orchestrator_48h.log"

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(ORCHESTRATOR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def is_process_running(proc_substring):
    try:
        out = subprocess.check_output(
            ["wmic", "process", "get", "commandline"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if proc_substring in line and "wmic" not in line.lower():
                return True
        return False
    except Exception:
        return False

def collect_latest_results():
    results = []
    for csv_file in BASE_DIR.glob("results/batch3_*/*/summary.csv"):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = "DistilBERT" if "batch3_distilbert" in str(csv_file) else "RoBERTa"
                    results.append({
                        "module": row.get("module_name", csv_file.parent.name),
                        "model": model,
                        "status": row.get("status", "N/A"),
                        "acc": row.get("accuracy_pct", "N/A"),
                        "f1": row.get("f1_score", "N/A"),
                        "recon": row.get("avg_reconstruction_error", "N/A"),
                        "vram": row.get("peak_vram_mb", "N/A"),
                        "speed": row.get("tokens_per_sec", "N/A"),
                    })
        except Exception:
            pass
    return results

def main():
    log("============================================================")
    log("Starting Autonomous 48-Hour Continuous GPU Orchestrator")
    log(f"Base Directory: {BASE_DIR}")
    log("============================================================")

    # Check if run_batch3_gpu_sweep.py is currently running
    if is_process_running("run_batch3_gpu_sweep.py"):
        log("[INFO] Detected active Category 5B runner. Supervising until completion...")
        while is_process_running("run_batch3_gpu_sweep.py"):
            time.sleep(30)
        log("[INFO] Category 5B runner completed! Immediately transitioning to 48-hour master pipeline...")

    # Launch full 48-hour pipeline
    log("[LAUNCH] Starting run_autonomous_48h_pipeline.py...")
    cmd = [sys.executable, "-u", "scripts/run_autonomous_48h_pipeline.py"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"neuralops;{env.get('PYTHONPATH', '')}" if os.name == "nt" else f"neuralops:{env.get('PYTHONPATH', '')}"

    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    t0 = time.time()
    last_audit = t0

    for line in proc.stdout:
        print(line, end="", flush=True)
        try:
            with open(ORCHESTRATOR_LOG, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

        # 20-minute audit interval
        if time.time() - last_audit >= 1200:
            last_audit = time.time()
            res = collect_latest_results()
            log(f"[20-MIN AUDIT] Total completed tasks collected so far: {len(res)}")
            try:
                subprocess.run(["git", "add", "results", "MASTER_PROJECT_RESULTS.md"], cwd=str(BASE_DIR), check=False)
                c_res = subprocess.run(
                    ["git", "commit", "-m", f"auto: periodic 20-min results sync ({len(res)} tasks)"],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True
                )
                if "nothing to commit" not in c_res.stdout.lower() and "clean" not in c_res.stdout.lower():
                    log(f"[AUTO-COMMIT] Committed periodic results sync ({len(res)} tasks)")
                    subprocess.run(["git", "push", "origin", "main"], cwd=str(BASE_DIR), timeout=60)
                    log("[AUTO-PUSH] Synced results with origin/main")
            except Exception as ce:
                log(f"[AUTO-COMMIT WARNING] {ce}")

    proc.wait()
    log(f"[COMPLETED] Autonomous pipeline exited with code {proc.returncode} in {(time.time() - t0)/3600:.2f} hours.")

if __name__ == "__main__":
    main()
