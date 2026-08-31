@echo off
cd /d D:\SVD_Project
set PYTHONPATH=D:\SVD_Project\Codes
echo STARTING_SWEEP_NOW > results\all_modules_paper_run\sweep_master.log
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --modules all --models distilbert --datasets sst2 --results_dir results/all_modules_paper_run --epochs 1 --batch_size 64 --timeout 600 >> results\all_modules_paper_run\sweep_master.log 2>&1
echo SWEEP_FINISHED >> results\all_modules_paper_run\sweep_master.log
