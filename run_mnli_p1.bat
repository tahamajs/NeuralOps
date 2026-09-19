@echo off
cd /d D:\svd_project
setlocal enabledelayedexpansion

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GIT_TERMINAL_PROMPT=0
set OMP_NUM_THREADS=1
set LOKY_MAX_CPU_COUNT=1

echo =====================================================================
echo [STAGE 2/3] MNLI Sweep - parallel=1 (maximum stability for 8GB VRAM)
echo [FIX 1] stat_margin_pct defined for all datasets
echo [FIX 2] Threshold calibrated to 36.90%% (majority 35.45%% + 1.45%% 3-sigma)
echo [FIX 3] micro_bs capped to 2
echo [FIX 4] Blocklist enforced
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
    --parallel 1 >> logs\distilbert_mnli_sweep.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MNLI sweep failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo =====================================================================
echo [STAGE 3/3] AG News Sweep - parallel=1
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
    --parallel 1 >> logs\distilbert_agnews_sweep.log 2>&1

echo =====================================================================
echo [DONE] MNLI + AG News completed successfully!
echo =====================================================================
