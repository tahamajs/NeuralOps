@echo off
cd /d "%~dp0"
start /b "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" Codes\framework\live_monitor.py > monitor.log 2>&1
start /b "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" Codes\framework\run_suite_multiprocess.py > suite.log 2>&1
