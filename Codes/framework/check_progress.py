import json
import subprocess
import time
import os
from pathlib import Path

LOG = Path("logs/progress_check.log")
STATE_FILE = Path("paper/_PROGRESS.md")
QUEUE_FILE = Path("paper/_PIPELINE_QUEUE.json")

def log(msg):
    # Ensure logs dir exists
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def check_daemon():
    try:
        out = subprocess.check_output([
            "powershell", "-Command",
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*auto_pipeline*' } | Select ProcessId"
        ]).decode()
        if "ProcessId" not in out:
            log("Daemon dead — restarting")
            subprocess.Popen('cmd.exe /c start /B "" "C:\\Users\\Hadi Schumacher\\anaconda3\\python.exe" Codes\\framework\\auto_pipeline.py > logs\\auto_pipeline.log 2>&1', shell=True)
    except Exception as e:
        log(f"Error checking daemon: {e}")

def check_gpu():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"
        ]).decode().strip()
        return out
    except Exception:
        return "Unknown"

def check_queue():
    try:
        if not QUEUE_FILE.exists():
            return 0, 0, 0, 0
        with open(QUEUE_FILE) as f:
            q = json.load(f)
        active = sum(1 for j in q.get("jobs", []) if j.get("status") == "active")
        pending = sum(1 for j in q.get("jobs", []) if j.get("status") == "pending")
        done = sum(1 for j in q.get("jobs", []) if j.get("status") == "done")
        fail = sum(1 for j in q.get("jobs", []) if j.get("status") == "failed")
        return active, pending, done, fail
    except Exception as e:
        log(f"Error checking queue: {e}")
        return 0, 0, 0, 0

def main():
    # Change cwd to project root assuming this script is run from project root or inside Codes/framework
    if Path("Codes/framework").exists():
        pass
    elif Path("../../Codes/framework").exists():
        os.chdir("../..")
        
    gpu = check_gpu()
    active, pending, done, fail = check_queue()
    log(f"GPU: {gpu} | active={active}/8 pending={pending} done={done} fail={fail}")
    
    check_daemon()
    
    with open(STATE_FILE, "w") as f:
        f.write(f"# Progress Check\n")
        f.write(f"**Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**GPU**: {gpu}\n")
        f.write(f"**Queue**: active={active}/8, pending={pending}, done={done}, failed={fail}\n")
        f.write(f"**Current Phase**: Running through queued jobs\n")

if __name__ == "__main__":
    main()
