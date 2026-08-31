@echo off
cd /d D:\SVD_Project
set PYTHONPATH=D:\SVD_Project\Codes;D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" Codes\framework\live_monitor.py --results_dir results\final_paper_run --port 8080 > results\final_paper_run\monitor_stdout.log 2>&1