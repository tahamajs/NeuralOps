@echo off
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\svd_project

echo =====================================================================
echo [STAGE 3/3] DistilBERT AG News Sweep from FULL Checkpoint (Safe Mode)
echo Model: models\distilbert_agnews_full
echo Concurrency: parallel=2, batch_size=16 (Zero OOM / High GPU efficiency)
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_agnews_full ^
  --model_family bert ^
  --dataset ag_news ^
  --dataset_dir D:\svd_project\Codes\framework\data\glue_data\ag_news ^
  --output_dir results\unified_sweep\distilbert_agnews_full ^
  --batch_size 16 ^
  --parallel 2 ^
  --finetune_steps 200 ^
  --recovery_steps 100 ^
  --learning_rate 2e-5 ^
  --precision fp16 >> logs\distilbert_agnews_full_sweep.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] AG News full sweep failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo =====================================================================
echo [DONE] AG News full sweep finished!
echo =====================================================================
