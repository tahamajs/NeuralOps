@echo off
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set PYTHONUNBUFFERED=1
set PYTHONPATH=D:\svd_project;D:\svd_project\Codes
cd /d D:\svd_project
python -u D:\svd_project\full_a3_sweep_runner.py >> D:\svd_project\results\a3_optimization\sweep_execution_v2.log 2>&1
