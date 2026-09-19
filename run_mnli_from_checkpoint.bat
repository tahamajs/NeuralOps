@echo off
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GIT_TERMINAL_PROMPT=0
set OMP_NUM_THREADS=1
set LOKY_MAX_CPU_COUNT=1
cd /d D:\svd_project

echo =====================================================================
echo [STAGE 2/3] DistilBERT MNLI Sweep from Fine-Tuned Checkpoint
echo Model: models\distilbert_mnli
echo Concurrency: parallel=1 (Single worker to eliminate paging crashes)
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_mnli ^
  --model_family bert ^
  --dataset mnli ^
  --dataset_dir Codes\framework\data\glue_data\mnli ^
  --output_dir results\unified_sweep\distilbert_mnli ^
  --batch_size 8 ^
  --parallel 1 ^
  --finetune_steps 200 ^
  --recovery_steps 100 ^
  --learning_rate 2e-5 ^
  --precision fp16 ^
  --fresh >> logs\distilbert_mnli_sweep.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MNLI sweep failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo =====================================================================
echo [STAGE 3/3] DistilBERT AG News Sweep
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py ^
  --model distilbert-base-uncased ^
  --model_family bert ^
  --dataset ag_news ^
  --dataset_dir Codes\framework\data\glue_data\ag_news ^
  --output_dir results\unified_sweep\distilbert_agnews ^
  --batch_size 8 ^
  --parallel 1 ^
  --finetune_steps 300 ^
  --recovery_steps 100 ^
  --learning_rate 2e-5 ^
  --precision fp16 >> logs\distilbert_agnews_sweep.log 2>&1

echo =====================================================================
echo All sweeps completed successfully!
echo =====================================================================
