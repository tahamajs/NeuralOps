@echo off
cd /d "D:\SVD_Project"
set PYTHONPATH=D:\SVD_Project\Codes
python -u Codes\framework\launch_gpu_server.py --mode comprehensive --precision fp16 --batch_size 1024 --seq_len 192 --max_steps 15 --timeout 120 --output_dir results\adaptive_v3 > D:\SVD_Project\adaptive_sweep_v3.log 2>&1
