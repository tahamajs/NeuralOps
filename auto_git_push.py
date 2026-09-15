import time
import subprocess
import os
import sys
from pathlib import Path

REPO_DIR = Path(r"D:\SVD_Project") if os.path.exists(r"D:\SVD_Project") else Path(__file__).resolve().parent

env = {
    **os.environ,
    "GIT_SSH_COMMAND": "ssh -o Port=443 -o HostName=ssh.github.com -o StrictHostKeyChecking=no",
}

print(f"[Auto-Git-Push] Service started for {REPO_DIR}. Sync interval: 60s", flush=True)

while True:
    try:
        # Always sync latest code from origin/main so the GPU server is always up to date
        try:
            subprocess.run(
                ["git", "pull", "--autostash", "--rebase", "origin", "main"],
                cwd=REPO_DIR,
                env=env,
                timeout=60,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        # Stage results documentation, json, csv, and state
        subprocess.run(["git", "add", "results/**/*.csv"], cwd=REPO_DIR, env=env, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "add", "results/**/*.json"], cwd=REPO_DIR, env=env, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "add", "results/**/*.md"], cwd=REPO_DIR, env=env, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "add", "finish_benchmark_state.json"], cwd=REPO_DIR, env=env, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "add", "results/*.csv"], cwd=REPO_DIR, env=env, stderr=subprocess.DEVNULL)

        # Untrack any log files if accidentally staged
        subprocess.run(["git", "rm", "--cached", "*.log", "results/**/console.log"], cwd=REPO_DIR, env=env, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            env=env,
        )

        staged_lines = [l for l in status.stdout.splitlines() if l.startswith(("M ", "A ", "AM", "MM"))]
        if staged_lines:
            subprocess.run(["git", "pull", "--autostash", "--rebase", "origin", "main"], cwd=REPO_DIR, env=env, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            commit_res = subprocess.run(
                ["git", "commit", "-m", "chore(results): auto-sync latest GPU benchmark results"],
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                env=env,
            )
            push_res = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            if push_res.returncode == 0:
                print(f"[Auto-Git-Push] Successfully pushed {len(staged_lines)} files to GitHub.", flush=True)
            else:
                print(f"[Auto-Git-Push] Push returned code {push_res.returncode}: {push_res.stderr.strip()}", flush=True)
                if "rejected" in push_res.stderr.lower():
                    subprocess.run(["git", "pull", "--autostash", "--rebase", "origin", "main"], cwd=REPO_DIR, env=env, timeout=60)
                    subprocess.run(["git", "push", "origin", "main"], cwd=REPO_DIR, env=env, timeout=60)
        else:
            print("[Auto-Git-Push] No staged results to push.", flush=True)

    except Exception as e:
        print(f"[Auto-Git-Push] Error: {e}", flush=True)

    time.sleep(60)
