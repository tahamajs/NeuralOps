#!/usr/bin/env python3
"""
Automated 15-minute background snapshot & git sync daemon.
Pulls latest logs from remote GPU server, parses clean summaries, and commits to GitHub.
"""

import datetime
import os
import re
import subprocess
import time

SNAPSHOT_DIR = "results/gpu_server/snapshot_2026_08_26"
SSH_CMD = [
    "ssh", "-o", "StrictHostKeyChecking=no", "-i",
    os.path.expanduser("~/Desktop/host_access"),
    "Hadi Schumacher@192.168.1.104"
]

def fetch_remote_files():
    # Pull raw log
    p1 = subprocess.run(
        SSH_CMD + ["powershell -Command \"Get-Content 'C:\\Users\\Hadi Schumacher\\SVD_Project\\comprehensive_run_detached.log'\""],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore"
    )
    raw_log = p1.stdout

    # Pull benchmark summary csv
    p2 = subprocess.run(
        SSH_CMD + ["powershell -Command \"Get-Content 'C:\\Users\\Hadi Schumacher\\SVD_Project\\results\\gpu_server\\benchmark_summary.csv'\""],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore"
    )
    raw_csv = p2.stdout

    return raw_log, raw_csv

def parse_log(raw_log_content):
    lines = raw_log_content.splitlines()
    parsed_rows = []
    pass_count = 0
    fail_count = 0
    timeout_count = 0
    timestamps = []

    for line in lines:
        line_clean = line.strip()
        if "->" in line_clean and any(s in line_clean for s in ["PASS", "FAIL", "TIMEOUT"]):
            ts_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line_clean)
            if ts_match:
                try:
                    dt = datetime.datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                    timestamps.append(dt)
                except Exception:
                    pass

            m = re.search(r'\[(track[0-9]_[a-zA-Z0-9_]+)\]\s+([a-zA-Z0-9_\-]+)\s+\|\s+([a-zA-Z0-9_\-]+)\s+\|\s+([a-zA-Z0-9_\-]+)\s+->\s+(\w+)\s+\((.*?)\)', line_clean)
            if m:
                track, module, model, dataset, status, details = m.groups()
                fwd = "-"
                bwd = "-"
                fwd_m = re.search(r'fwd:\s*([\d\.]+)ms', details)
                bwd_m = re.search(r'bwd:\s*([\d\.]+)ms', details)
                if fwd_m:
                    fwd = f"{fwd_m.group(1)} ms"
                if bwd_m:
                    bwd = f"{bwd_m.group(1)} ms"

                if status == "PASS":
                    pass_count += 1
                elif status == "TIMEOUT":
                    timeout_count += 1
                else:
                    fail_count += 1

                parsed_rows.append({
                    "track": track,
                    "module": module,
                    "model": model.upper(),
                    "dataset": dataset.upper(),
                    "status": status,
                    "fwd": fwd,
                    "bwd": bwd,
                })
        elif "TIMEOUT:" in line_clean and "exceeded" in line_clean:
            ts_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line_clean)
            if ts_match:
                try:
                    dt = datetime.datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                    timestamps.append(dt)
                except Exception:
                    pass

            m = re.search(r'TIMEOUT:\s+([a-zA-Z0-9_\-]+)\s+on\s+([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)\s+exceeded', line_clean)
            if m:
                module, model, dataset = m.groups()
                timeout_count += 1
                parsed_rows.append({
                    "track": "track1_svd",
                    "module": module,
                    "model": model.upper(),
                    "dataset": dataset.upper(),
                    "status": "TIMEOUT",
                    "fwd": "TIMEOUT (>120s)",
                    "bwd": "TIMEOUT (>120s)",
                })

    return parsed_rows, pass_count, fail_count, timeout_count, timestamps

def run_snapshot_cycle():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    raw_log, raw_csv = fetch_remote_files()

    if raw_csv and len(raw_csv.strip()) > 0:
        with open(os.path.join(SNAPSHOT_DIR, "benchmark_summary.csv"), "w", encoding="utf-8") as f:
            f.write(raw_csv)

    rows, pass_c, fail_c, to_c, ts = parse_log(raw_log)
    total_completed = len(rows)

    summary_md = f"""# GPU Server Benchmark: Partial Sweep Snapshot Summary

**Snapshot Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Completed Combinations:** {total_completed} / 720  
**Status Breakdown:**
- **PASS:** {pass_c}
- **TIMEOUT:** {to_c}
- **FAIL:** {fail_c}

---

## Execution Results Table

| Track | Module | Model | Dataset | Status | Fwd Time | Bwd Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for r in rows:
        summary_md += f"| `{r['track']}` | `{r['module']}` | {r['model']} | {r['dataset']} | **{r['status']}** | {r['fwd']} | {r['bwd']} |\n"

    with open(os.path.join(SNAPSHOT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    readme_md = f"""# Partial GPU Sweep Snapshot (2026-08-26)

> **DISCLAIMER:** This is a partial snapshot of an ongoing sweep, not the final benchmark result.

### Snapshot Metadata
* **Last Snapshot Update:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
* **Hardware Device:** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM, Compute Capability 8.9)
* **Hyperparameters:** `batch_size=128`, `seq_len=512`, `precision=bf16`, `max_steps=5`, `timeout=120s`
* **Sweep Progress:** {total_completed} / 720 combinations completed ({pass_c} PASS, {to_c} TIMEOUT, {fail_c} FAIL)

### Included Files
* `summary.md`: Clean, parsed execution table and status breakdown.
* `benchmark_summary.csv`: Machine-readable CSV metrics.
"""
    with open(os.path.join(SNAPSHOT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_md)

    # Git stage, commit, push
    subprocess.run(["git", "add", "-f", SNAPSHOT_DIR], check=True)
    status_out = subprocess.run(["git", "status", "--porcelain"], stdout=subprocess.PIPE, text=True).stdout
    if status_out.strip():
        subprocess.run(["git", "commit", "-m", f"chore(benchmark): auto-snapshot update ({total_completed}/720 combos)"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auto-snapshot committed and pushed: {total_completed}/720 combos.")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No new combos to commit ({total_completed}/720).")

if __name__ == "__main__":
    print("Starting Auto-Snapshot daemon (15-minute loop)...")
    while True:
        try:
            run_snapshot_cycle()
        except Exception as e:
            print(f"Error in snapshot cycle: {e}")
        time.sleep(900)
