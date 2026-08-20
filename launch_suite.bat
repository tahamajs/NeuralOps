@echo off
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
cd /d "C:\Users\Hadi Schumacher\SVD_Project\Codes\framework"
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u run_suite_multiprocess.py > suite.log 2>&1
