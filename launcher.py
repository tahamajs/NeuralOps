import subprocess
import os

python_exe = r"C:\Users\Hadi Schumacher\anaconda3\python.exe"
cwd = r"D:\SVD_Project"

def launch(cmd_list, out_file, err_file):
    out = open(os.path.join(cwd, out_file), 'w')
    err = open(os.path.join(cwd, err_file), 'w')
    p = subprocess.Popen(cmd_list, cwd=cwd, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    print(f"Launched {cmd_list[1]} with PID {p.pid}")

launch([python_exe, "scripts/run_monitor_server.py"], "results/monitor.log", "results/monitor_err.log")
launch([python_exe, "auto_upload_hf_results.py", "180"], "results/uploader.log", "results/uploader_err.log")
launch([python_exe, "-u", "run_all_benchmarks_queue.py"], "results/queue_master.log", "results/queue_master_err.log")
