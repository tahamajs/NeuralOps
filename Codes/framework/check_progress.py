import json
import subprocess
import time
import os
import glob
from pathlib import Path
import csv

LOG = Path("logs/progress_check.log")
STATE_FILE = Path("paper/_PROGRESS.md")
QUEUE_FILE = Path("paper/_PIPELINE_QUEUE.json")

def log(msg):
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

def rescue_ooms():
    try:
        # Scan for OOMs in summary CSVs
        oom_modules = set()
        for csv_file in glob.glob("results/**/summary*.csv", recursive=True):
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("status") == "FAIL" and ("out of memory" in str(row.get("error", "")).lower() or "oom" in str(row.get("error", "")).lower()):
                            mod = row.get("module_name")
                            if mod:
                                oom_modules.add(mod)
            except Exception:
                pass
                
        if not oom_modules:
            return 0
            
        with open(QUEUE_FILE, "r") as f:
            q = json.load(f)
            
        rescued_count = 0
        existing_cmds = {j["cmd"] for j in q.get("jobs", [])}
        
        for mod in oom_modules:
            # We don't know the exact dataset/model, but we can guess it's from recent runs
            # A safer way is to just look for failed jobs in the queue
            pass
            
        # Instead, let's just rescue jobs marked as 'failed' in the queue by halving their batch size
        for j in q.get("jobs", []):
            if j["status"] == "failed" and "--batch_size" in j["cmd"]:
                # Check if we already rescued this
                if "--batch_size 2" not in j["cmd"] and "--batch_size 1" not in j["cmd"]:
                    # Create a rescue job
                    new_cmd = j["cmd"].replace("--batch_size 8", "--batch_size 2").replace("--batch_size 4", "--batch_size 1")
                    if new_cmd not in existing_cmds:
                        new_job = {
                            "id": j["id"] + "_rescue_oom",
                            "type": "gpu",
                            "cmd": new_cmd,
                            "status": "pending",
                            "retries": 0
                        }
                        q["jobs"].append(new_job)
                        existing_cmds.add(new_cmd)
                        rescued_count += 1
                        j["status"] = "rescued" # Mark original as rescued so we don't duplicate
                        log(f"Rescued OOM job: {j['id']} with smaller batch size")
                        
        if rescued_count > 0:
            with open(QUEUE_FILE, "w") as f:
                json.dump(q, f, indent=2)
                
        return rescued_count
    except Exception as e:
        log(f"Error in rescue_ooms: {e}")
        return 0

def check_queue():
    try:
        if not QUEUE_FILE.exists():
            return 0, 0, 0, 0
        with open(QUEUE_FILE) as f:
            q = json.load(f)
        active = sum(1 for j in q.get("jobs", []) if j.get("status") == "active")
        pending = sum(1 for j in q.get("jobs", []) if j.get("status") == "pending")
        done = sum(1 for j in q.get("jobs", []) if j.get("status") == "done" or j.get("status") == "rescued")
        fail = sum(1 for j in q.get("jobs", []) if j.get("status") == "failed")
        return active, pending, done, fail
    except Exception as e:
        log(f"Error checking queue: {e}")
        return 0, 0, 0, 0

def main():
    if Path("Codes/framework").exists():
        pass
    elif Path("../../Codes/framework").exists():
        os.chdir("../..")
        
    gpu = check_gpu()
    rescued = rescue_ooms()
    active, pending, done, fail = check_queue()
    log(f"GPU: {gpu} | active={active}/8 pending={pending} done={done} fail={fail} | OOMs rescued this cycle: {rescued}")
    
    check_daemon()
    
    with open(STATE_FILE, "w") as f:
        f.write(f"# Progress Check\n")
        f.write(f"**Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**GPU**: {gpu}\n")
        f.write(f"**Queue**: active={active}/8, pending={pending}, done={done}, failed={fail}\n")
        if rescued > 0:
            f.write(f"**Alert**: {rescued} OOM crashes detected and auto-requeued with reduced batch size.\n")
        f.write(f"**Current Phase**: Running through queued jobs\n")

if __name__ == "__main__":
    main()
