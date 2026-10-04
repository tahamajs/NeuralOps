@echo off
echo Starting Unified Master Pipeline for all KD survivors...
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=.

python scripts\queue_unified_runs.py --batch_size 16

echo All Unified Benchmarks Complete!
pause
