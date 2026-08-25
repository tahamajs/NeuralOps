@echo off
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=C:\Users\Hadi Schumacher\SVD_Project\Codes
python Codes\framework\launch_gpu_server.py --mode fast --precision fp16 --batch_size 16 --output_dir results\gpu_server > gpu_run.log 2>&1
