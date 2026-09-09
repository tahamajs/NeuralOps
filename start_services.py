import subprocess
import os
import sys

cwd = r"D:\SVD_Project"
python_exe = r"C:\Users\Hadi Schumacher\anaconda3\python.exe"

def run_bg(args, out_path):
    with open(out_path, 'w') as f:
        # DETACHED_PROCESS = 0x00000008
        p = subprocess.Popen(args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, creationflags=0x00000008)
        print(f"Launched {' '.join(args)} with PID {p.pid}")

run_bg([python_exe, "-u", "scripts/run_monitor_server.py"], os.path.join(cwd, r"results\mon2.log"))
run_bg([python_exe, "-u", "auto_upload_hf_results.py", "180"], os.path.join(cwd, r"results\up2.log"))
run_bg([python_exe, "-u", "run_all_benchmarks_queue.py"], os.path.join(cwd, r"results\master_q2.log"))
