@echo off
cd /d D:\svd_project
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GIT_TERMINAL_PROMPT=0
python -u Codes\framework\run_quality_sweep.py --model models\distilbert_sst2 --model_family bert --dataset sst2 --dataset_dir Codes\framework\data\glue_data\sst2 --output_dir results\unified_sweep\distilbert --batch_size 8 --recovery_steps 50 --timeout 1200 --precision fp16 --parallel 4 >> logs\distilbert_sweep.log 2>&1
