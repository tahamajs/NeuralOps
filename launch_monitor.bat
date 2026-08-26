@echo off
cd /d "D:\SVD_Project"
set PYTHONPATH=D:\SVD_Project\Codes
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\live_monitor.py > live_monitor_final.log 2>&1
