@echo off
cd /d D:\SVD_Project
set PYTHONPATH=D:\SVD_Project\Codes;D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" Codes\framework\run_quality_sweep.py --models distilbert --datasets sst2 --seeds 42 --pretrained --epochs 1 --batch_size 64 --output_dir results\final_paper_run --timeout 300 > results\final_paper_run\sweep_stdout.log 2>&1