@echo off
cd /d "D:\SVD_Project"
set PYTHONPATH=D:\SVD_Project\Codes
python -u Codes\framework\launch_gpu_server.py --mode comprehensive --precision fp16 --batch_size 1024 --seq_len 128 --max_steps 10 --timeout 120 --output_dir results\full_validation > D:\SVD_Project\full_sweep.log 2>&1
