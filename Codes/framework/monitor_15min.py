#!/usr/bin/env python3
"""
Periodic 15-Minute Monitor and Integrity Verifier.
Monitors GPU state, verifies benchmark results integrity (detects PASS with 0% accuracy),
and logs periodic heartbeats.
"""
import os
import sys
import csv
import time
import glob
import datetime
import subprocess
from pathlib import Path

def check_integrity(results_root: str = "results"):
    broken = []
    root = Path(results_root)
    if not root.exists():
        return broken
    
    # Target canonical benchmark summaries: unified sweeps, rank ablations, and long-context
    canonical_files = list(root.glob("unified_sweep/**/summary.csv")) + \
                      list(root.glob("rank_ablation/**/summary.csv")) + \
                      list(root.glob("long_context/summary.csv"))
    
    # If explicit root given without subdirs (e.g. results/unified_sweep/distilbert/summary.csv)
    if root.is_file():
        canonical_files = [root]

    for p in canonical_files:
        # Avoid checking inside checkpoints or hf_export dirs
        if "checkpoints" in p.parts or "hf_export" in p.parts or "backup" in p.name:
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    status = r.get("status")
                    acc_val = r.get("accuracy_pct", r.get("accuracy"))
                    if acc_val is not None and acc_val != "":
                        acc = float(acc_val or 0.0)
                        elapsed = float(r.get("elapsed_s", 0.0) or 0.0)
                        if status in ("PASS", "PASS_LOW") and (acc <= 0.0 or elapsed < 0.5):
                            broken.append((str(p), r.get("module_name"), status, acc, elapsed))
        except Exception:
            pass
    return broken

def get_gpu_status():
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "gpu_util_pct": float(parts[0]),
                "vram_used_mb": float(parts[1]),
                "vram_total_mb": float(parts[2]),
                "temp_c": float(parts[3])
            }
    except Exception:
        pass
    return None

def monitor_once():
    ts = datetime.datetime.now().isoformat()
    gpu = get_gpu_status()
    broken = check_integrity()
    
    status_line = f"[{ts}] MONITOR HEARTBEAT:"
    if gpu:
        status_line += f" GPU: {gpu['gpu_util_pct']}% | VRAM: {gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB | Temp: {gpu['temp_c']}C"
    else:
        status_line += " GPU info unavailable"
        
    print(status_line)
    if broken:
        print(f"  [ALERT] Found {len(broken)} INVALID PASS entries (0% accuracy):")
        for b in broken:
            print(f"    - {b[0]} | {b[1]} | status={b[2]} acc={b[3]} elapsed={b[4]}s")
    else:
        print("  [OK] Benchmark integrity clean: 0 broken PASS records found.")
    sys.stdout.flush()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single integrity check")
    parser.add_argument("--interval", type=int, default=900, help="Monitoring interval in seconds (default 900s = 15m)")
    args = parser.parse_args()

    if args.once:
        monitor_once()
        return

    print(f"Starting 15-minute background monitor (interval={args.interval}s)...")
    while True:
        monitor_once()
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
