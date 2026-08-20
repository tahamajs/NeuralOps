import urllib.request, json, subprocess
from pathlib import Path

EXPERIMENTS_DIR = Path(r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\experiments")
MASTER_LOG = Path(r"C:\Users\Hadi Schumacher\SVD_Project\trajectory_master.log")

print("==============================")
print("   FULL SYSTEM VERIFICATION   ")
print("==============================")

# 1. API + GPU + Training state
try:
    data = json.loads(urllib.request.urlopen("http://localhost:8080/api/stats").read().decode("utf-8"))
    g = data.get("gpu", {})
    t = data.get("train", {})
    print("[OK] API Server:  CONNECTED")
    print(f"[OK] GPU Load:    {g.get('gpu_util','?')}%")
    print(f"[OK] VRAM:        {round(g.get('mem_used_mb',0)/1024,1)} / {round(g.get('mem_total_mb',0)/1024,1)} GB")
    print(f"[OK] Temp/Power:  {g.get('temp_c','?')} C  /  {g.get('power_w','?')} W")
    ram_u = g.get("sys_ram_used_gb","?")
    ram_t = g.get("sys_ram_total_gb","?")
    print(f"[OK] System RAM:  {ram_u} / {ram_t} GB")
    print(f"[OK] Run Index:   {t.get('active_index','?')}")
    print(f"[OK] Active Model:{t.get('active_model','?')} / Task: {t.get('active_task','?')}")
    lh = t.get("loss_history",[])
    if lh:
        latest = lh[-1]
        print(f"[OK] Latest Loss: {latest.get('loss','?')} @ step {latest.get('step','?')}")
    matrix = data.get("matrix",[])
    completed = sum(1 for task_data in matrix for r in task_data.get("runs",[]) if r.get("status")=="completed")
    total_m = sum(len(task_data.get("runs",[])) for task_data in matrix)
    print(f"[OK] UI Matrix:   {completed}/{total_m} completed")
except Exception as e:
    print(f"[ERR] API: {e}")

# 2. Process check
r = subprocess.run(["powershell","-Command","(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_suite' }).ProcessId"], capture_output=True, text=True)
pid = r.stdout.strip()
print(f"[OK] run_suite.py PID: {pid}" if pid else "[ERR] run_suite.py NOT RUNNING!")

r2 = subprocess.run(["powershell","-Command","(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'live_monitor' }).ProcessId"], capture_output=True, text=True)
pid2 = r2.stdout.strip()
print(f"[OK] live_monitor PID: {pid2}" if pid2 else "[ERR] live_monitor.py NOT RUNNING!")

# 3. Experiments
n = len(list(EXPERIMENTS_DIR.glob("*"))) if EXPERIMENTS_DIR.exists() else 0
print(f"[OK] Experiments: {n} folders saved to disk")

# 4. Master log
if MASTER_LOG.exists():
    lines = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    success = sum(1 for l in lines if "<< SUCCESS" in l)
    errors  = sum(1 for l in lines if "NOTICE:" in l and "Error:" in l)
    skipped = sum(1 for l in lines if "SKIPPING:" in l)
    print(f"[OK] Master Log:  {success} SUCCESS | {errors} errors | {skipped} skipped")
else:
    print("[ERR] trajectory_master.log NOT found!")

print("==============================")
