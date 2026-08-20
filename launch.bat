set PYTHONPATH=C:\Users\Hadi Schumacher\SVD_Project\Codes
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Hadi Schumacher\SVD_Project\Codes\framework"
start /B "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" run_suite.py > suite.log 2>&1
start /B "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" live_monitor.py > monitor.log 2>&1
