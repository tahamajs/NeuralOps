@echo off
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set HF_ENDPOINT=https://hf.devneeds.ir/
set PYTHONPATH=C:\Users\Hadi Schumacher\SVD_Project\Codes
python Codes\framework\launch_gpu_server.py --mode comprehensive --batch_size 128 --seq_len 512 > comprehensive_run_detached.log 2>&1
