#!/usr/bin/env python3
"""
Background Auto-Committer for Results and Metrics to Git.
Periodically stages summary CSVs, metric JSONs, and code updates,
commits them with informative messages, and pushes to origin/main.
"""

import subprocess
import time
import sys
from datetime import datetime

CHECK_INTERVAL_S = 300  # Every 5 minutes

def run_cmd(cmd, cwd="D:\\SVD_Project"):
    res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()

def sync_git():
    # 1. Check for changes in results and codes
    code, stdout, _ = run_cmd("git status --porcelain results/ Codes/")
    if not stdout:
        return

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detected new results/code changes. Staging...")

    # Stage results CSVs, JSONs, PNG plots, and modified codes
    run_cmd("git add results/**/*.csv results/**/*.json results/**/*.png Codes/framework/")

    # Check if there is anything actually staged
    code, staged, _ = run_cmd("git diff --cached --name-only")
    if not staged:
        print("No metrics/code files staged (only ignored or log files changed).")
        return

    staged_files = staged.split("\n")
    print(f"Staged {len(staged_files)} files. Committing...")

    commit_msg = f"auto-sync: update benchmark metrics and results [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
    code, out, err = run_cmd(f'git commit -m "{commit_msg}"')
    if code != 0:
        print(f"Commit failed or clean: {err}")
        return

    print("Commit created successfully. Pushing to origin main...")
    code, out, err = run_cmd("git push origin main")
    if code == 0:
        print("Pushed to GitHub successfully!")
    else:
        print(f"Push warning/error: {err}")

def loop(interval=CHECK_INTERVAL_S):
    print(f"Starting Auto-Git-Sync daemon (interval: {interval}s)...")
    while True:
        try:
            sync_git()
        except Exception as e:
            print(f"Error in sync_git: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else CHECK_INTERVAL_S
    loop(interval)
