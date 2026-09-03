@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo Universal Model SST-2 Evaluation Suite (Max 120s Timeout Per Module)
echo ======================================================================

set PYTHONPATH=D:\SVD_Project\Codes;%PYTHONPATH%
set DATA_DIR=D:\SVD_Project\Codes\framework\data\glue_data\sst2

echo.
echo [1/4] Running GPT-2 on SST-2...
python D:\SVD_Project\Codes\framework\run_quality_sweep.py ^
  --model D:\SVD_Project\models\gpt2 ^
  --model_family gpt ^
  --dataset sst2 ^
  --dataset_dir %DATA_DIR% ^
  --output_dir D:\SVD_Project\results\gpt2_sst2_final_paper_run ^
  --batch_size 16 ^
  --recovery_steps 10 ^
  --finetune_steps 20 ^
  --timeout 120 ^
  --precision fp16

echo.
echo [2/4] Running RoBERTa-base on SST-2...
python D:\SVD_Project\Codes\framework\run_quality_sweep.py ^
  --model D:\SVD_Project\models\roberta-base ^
  --model_family roberta ^
  --dataset sst2 ^
  --dataset_dir %DATA_DIR% ^
  --output_dir D:\SVD_Project\results\roberta_sst2_final_paper_run ^
  --batch_size 16 ^
  --recovery_steps 10 ^
  --finetune_steps 20 ^
  --timeout 120 ^
  --precision fp16

echo.
echo [3/4] Running DistilBERT on SST-2...
python D:\SVD_Project\Codes\framework\run_quality_sweep.py ^
  --model D:\SVD_Project\models\distilbert_sst2 ^
  --model_family bert ^
  --dataset sst2 ^
  --dataset_dir %DATA_DIR% ^
  --output_dir D:\SVD_Project\results\distilbert_sst2_final_paper_run ^
  --batch_size 16 ^
  --recovery_steps 10 ^
  --finetune_steps 20 ^
  --timeout 120 ^
  --precision fp16

echo.
echo [4/4] Running Qwen3.5-0.8B on SST-2...
python D:\SVD_Project\Codes\framework\run_quality_sweep.py ^
  --model D:\SVD_Project\models\Qwen3.5-0.8B ^
  --model_family qwen3_5 ^
  --dataset sst2 ^
  --dataset_dir %DATA_DIR% ^
  --output_dir D:\SVD_Project\results\qwen3_5_sst2_final_paper_run ^
  --batch_size 16 ^
  --recovery_steps 10 ^
  --finetune_steps 50 ^
  --timeout 120 ^
  --precision fp16

echo.
echo ======================================================================
echo All Model Sweeps Completed Successfully!
echo ======================================================================
