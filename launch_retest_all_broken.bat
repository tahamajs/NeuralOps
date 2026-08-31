@echo off
setlocal

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

cd /d D:\SVD_Project

python -u retest_all_broken.py > D:\SVD_Project\retest_all_broken.log 2>&1

endlocal
