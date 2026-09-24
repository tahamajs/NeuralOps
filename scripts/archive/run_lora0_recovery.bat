@echo off
setlocal
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\svd_project

set /p LORA_MODULES=<D:\svd_project\results\lora_only_modules.txt

echo =====================================================================
echo [PHASE 1] Recovering 51 LoRA-only Modules with --lora_rank 0 (PARALLEL=3)
echo =====================================================================

echo [1/3] Running on MNLI...
python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_mnli_full ^
  --dataset mnli ^
  --dataset_dir Codes\framework\data\glue_data\mnli ^
  --modules %LORA_MODULES% ^
  --lora_rank 0 --batch_size 8 --parallel 2 --no_skip ^
  >> logs\lora0_mnli.log 2>&1

echo [2/3] Running on SST-2 (PARALLEL=3)...
python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_sst2 ^
  --dataset sst2 ^
  --dataset_dir Codes\framework\data\glue_data\sst2 ^
  --modules %LORA_MODULES% ^
  --lora_rank 0 --batch_size 8 --parallel 3 --no_skip ^
  >> logs\lora0_sst2.log 2>&1

echo [3/3] Running on AG News (PARALLEL=3)...
python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_agnews_full ^
  --dataset ag_news ^
  --dataset_dir Codes\framework\data\glue_data\ag_news ^
  --modules %LORA_MODULES% ^
  --lora_rank 0 --batch_size 8 --parallel 3 --no_skip ^
  >> logs\lora0_agnews.log 2>&1

echo =====================================================================
echo ALL LORA-0 RUNS COMPLETED SUCCESSFULLY!
echo =====================================================================
