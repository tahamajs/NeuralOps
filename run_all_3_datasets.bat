@echo off
cd /d D:\svd_project
setlocal enabledelayedexpansion

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GIT_TERMINAL_PROMPT=0

echo =====================================================================
echo [STAGE 1/3] Running DistilBERT SST-2 Sweep (parallel=4)
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py --model models\distilbert_sst2 --model_family bert --dataset sst2 --dataset_dir Codes\framework\data\glue_data\sst2 --output_dir results\unified_sweep\distilbert --batch_size 8 --recovery_steps 50 --timeout 1200 --precision fp16 --parallel 4 >> logs\distilbert_sst2_sweep.log 2>&1

echo =====================================================================
echo [STAGE 2/3] Running DistilBERT MNLI (3-Class) Sweep (parallel=4)
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py --model distilbert-base-uncased --model_family bert --dataset mnli --dataset_dir Codes\framework\data\glue_data\mnli --output_dir results\unified_sweep\distilbert_mnli --batch_size 8 --recovery_steps 50 --finetune_steps 300 --timeout 1200 --precision fp16 --parallel 4 >> logs\distilbert_mnli_sweep.log 2>&1

echo =====================================================================
echo [STAGE 3/3] Running DistilBERT AG_NEWS (4-Class) Sweep (parallel=4)
echo =====================================================================
python -u Codes\framework\run_quality_sweep.py --model distilbert-base-uncased --model_family bert --dataset ag_news --dataset_dir Codes\framework\data\glue_data\ag_news --output_dir results\unified_sweep\distilbert_agnews --batch_size 8 --recovery_steps 50 --finetune_steps 300 --timeout 1200 --precision fp16 --parallel 4 >> logs\distilbert_agnews_sweep.log 2>&1

echo =====================================================================
echo All 3 Datasets Completed Successfully!
echo =====================================================================
