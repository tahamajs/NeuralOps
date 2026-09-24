@echo off
setlocal
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\svd_project

set /p LORA_MODULES=<D:\svd_project\results\lora_only_modules.txt

echo Running SST-2 recovery sweep with parallel 3...
python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_sst2 ^
  --dataset sst2 ^
  --dataset_dir Codes\framework\data\glue_data\sst2 ^
  --modules %LORA_MODULES% ^
  --lora_rank 0 --batch_size 8 --parallel 3 --no_skip ^
  > logs\lora0_sst2.log 2>&1

echo SST-2 SWEEP FINISHED
