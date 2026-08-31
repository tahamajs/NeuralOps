
import subprocess
import sys
import os
import time

print("RUNNING SYNC AND PIPING")
os.chdir(r"D:\SVD_Project")
with open(r"results\all_modules_paper_run\sweep_master.log", "w") as f:
    f.write("LAUNCHING SYNC...\n")

with open(r"results\all_modules_paper_run\sweep_master.log", "a") as f:
    proc = subprocess.Popen(
        [
            r"C:\Users\Hadi Schumacher\anaconda3\python.exe",
            "-u",
            r"Codes\framework\run_quality_sweep.py",
            "--modules", "all",
            "--models", "distilbert",
            "--datasets", "sst2",
            "--results_dir", "results/all_modules_paper_run",
            "--epochs", "1",
            "--batch_size", "64",
            "--timeout", "600"
        ],
        stdout=f,
        stderr=subprocess.STDOUT
    )
    print(f"SPAWNED PID: {proc.pid}")
    # Don't exit, just hold it open so ssh keeps it alive
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
