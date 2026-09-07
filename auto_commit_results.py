#!/usr/bin/env python3
"""
Background Auto-Committer for Benchmark Results.

Periodically monitors the results/ directory and commits new/updated
metrics files (summary.csv, *metrics.json, charts, summaries) to git
and pushes them to GitHub automatically without blocking the benchmark runner.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list) -> tuple:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return -1, "", str(exc)


def auto_commit_loop(interval_s: int = 600):
    repo_dir = Path(os.environ.get("SVD_PROJECT_ROOT", "D:\\SVD_Project" if os.path.exists("D:\\SVD_Project") else "."))
    os.chdir(repo_dir)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auto-committer started in {repo_dir.resolve()}. Checking every {interval_s}s...")

    while True:
        try:
            # Stage only metrics, csvs, jsons, pngs, and doc updates - ignore checkpoints and weights
            run_cmd(["git", "add", "results/**/*.csv", "results/**/*.json", "results/**/*.png", "Docs/content/**/*.md", "results/master_queue_benchmark_summary*"])

            # Check if there are staged changes
            ret, out, _ = run_cmd(["git", "diff", "--staged", "--name-only"])
            if ret == 0 and out.strip():
                staged_files = [f for f in out.splitlines() if f.strip()]
                commit_msg = f"chore(benchmark): auto-sync {len(staged_files)} benchmark results and metrics [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
                c_ret, c_out, c_err = run_cmd(["git", "commit", "-m", commit_msg])
                if c_ret == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Committed {len(staged_files)} files: {commit_msg}")
                    # Push to origin
                    p_ret, p_out, p_err = run_cmd(["git", "push", "origin", "main"])
                    if p_ret == 0:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully pushed latest results to GitHub origin/main.")
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Push failed (will retry next interval): {p_err or p_out}")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Commit issue: {c_err}")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No new benchmark metrics to commit yet.")

        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in auto-commit loop: {e}")

        time.sleep(interval_s)


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    auto_commit_loop(interval)
