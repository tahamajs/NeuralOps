@echo off
cd /d D:\svd_project
setlocal enabledelayedexpansion

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GIT_TERMINAL_PROMPT=0
REM Limit torch DLL load concurrency to avoid WinError 1455 paging file exhaustion
set OMP_NUM_THREADS=1
set LOKY_MAX_CPU_COUNT=2

echo =====================================================================
echo [STAGE 2/3] MNLI Sweep - parallel=2 (safe for 8GB VRAM)
echo [FIX] Reduced from 4 to 2 workers to prevent WinError 1455
echo [FIX] Blocklist loaded from Codes\framework\config\blocklist.txt
echo [FIX] PASS_LOW modules will be re-evaluated (only PASS is skipped)
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py ^
    --model distilbert-base-uncased ^
    --model_family bert ^
    --dataset mnli ^
    --dataset_dir Codes\framework\data\glue_data\mnli ^
    --output_dir results\unified_sweep\distilbert_mnli ^
    --batch_size 8 ^
    --recovery_steps 50 ^
    --finetune_steps 300 ^
    --timeout 1200 ^
    --precision fp16 ^
    --parallel 2 >> logs\distilbert_mnli_sweep.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MNLI sweep failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo =====================================================================
echo [STAGE 3/3] AG News Sweep - parallel=2
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py ^
    --model distilbert-base-uncased ^
    --model_family bert ^
    --dataset ag_news ^
    --dataset_dir Codes\framework\data\glue_data\ag_news ^
    --output_dir results\unified_sweep\distilbert_agnews ^
    --batch_size 8 ^
    --recovery_steps 50 ^
    --finetune_steps 300 ^
    --timeout 1200 ^
    --precision fp16 ^
    --parallel 2 >> logs\distilbert_agnews_sweep.log 2>&1

echo =====================================================================
echo [DONE] MNLI + AG News complete!
echo =====================================================================
