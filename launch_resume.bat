@echo off
setlocal

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

cd /d D:\SVD_Project

REM Ensure results directory exists and pre-populate with existing summary if present
if not exist "results\run_quality_resume" mkdir "results\run_quality_resume"
if not exist "results\run_quality_resume\summary.csv" (
    if exist "results\run_quality_all_screening\summary.csv" (
        copy "results\run_quality_all_screening\summary.csv" "results\run_quality_resume\summary.csv"
    )
)

python -u Codes\framework\run_quality_sweep.py ^
  --modules all_modules.txt ^
  --models distilbert ^
  --datasets sst2 ^
  --seeds 42 ^
  --epochs 1 ^
  --max_steps 100 ^
  --batch_size 128 ^
  --seq_len 128 ^
  --timeout 300 ^
  --precision fp16 ^
  --pretrained ^
  --gradient_checkpointing ^
  --output_dir results/run_quality_resume ^
  >> D:\SVD_Project\run_quality_resume.log 2>&1

endlocal
