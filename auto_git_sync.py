#!/usr/bin/env python3
"""
Bidirectional Git Auto-Sync Daemon.
Periodically commits local artifacts/logs and pulls/pushes with origin main.
Safely ignores conflict prone temporary files while preserving checkpoints and results.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

INTERVAL_SECONDS = 300  # Sync every 5 minutes

def run_cmd(cmd, cwd=None):
    try:
        res = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=120)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def sync_repo(repo_path: Path):
    if not (repo_path / ".git").exists():
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 1. Fetch remote updates
    rc, out, err = run_cmd("git fetch origin main", cwd=str(repo_path))
    
    # 2. Check for local modifications
    rc, status_out, _ = run_cmd("git status --porcelain", cwd=str(repo_path))
    if status_out:
        print(f"[{ts}] Local changes detected, staging...")
        run_cmd("git add -A papers/ results/ logs/ audit/ *.log *.csv *.json *.jsonl", cwd=str(repo_path))
        commit_msg = f"Auto-sync progress checkpoint [{ts}]"
        rc, _, _ = run_cmd(f'git commit -m "{commit_msg}"', cwd=str(repo_path))

    # 3. Pull with rebase to avoid merge conflicts
    rc, pull_out, pull_err = run_cmd("git pull --rebase origin main", cwd=str(repo_path))
    if rc != 0:
        print(f"[{ts}] Rebase failed, aborting: {pull_err}")
        run_cmd("git rebase --abort", cwd=str(repo_path))
    else:
        # 4. Push local changes
        rc, push_out, push_err = run_cmd("git push origin main", cwd=str(repo_path))
        if rc == 0:
            print(f"[{ts}] Git sync push success.")
        else:
            print(f"[{ts}] Git push skipped or non-fatal: {push_err}")

def main():
    repo_path = Path("D:/SVD_Project") if os.path.exists("D:/SVD_Project") else Path(__file__).resolve().parent
    print(f"Starting bidirectional Git Auto-Sync on {repo_path} (every {INTERVAL_SECONDS}s)...", flush=True)
    while True:
        try:
            sync_repo(repo_path)
        except Exception as e:
            print(f"Error in sync iteration: {e}", flush=True)
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
