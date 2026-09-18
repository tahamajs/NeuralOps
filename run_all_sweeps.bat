@echo off
cd /d D:\svd_project

echo "Starting DistilBERT Sweep"
python -u Codes\framework\run_quality_sweep.py --model models\distilbert_sst2 --model_family bert --dataset sst2 --dataset_dir Codes\framework\data\glue_data\sst2 --output_dir results\unified_sweep\distilbert --batch_size 16 --recovery_steps 50 --timeout 1200 --precision fp16 --parallel 2 --fresh > logs\distilbert_sweep.log 2>&1

echo "Starting RoBERTa Sweep"
python -u Codes\framework\run_quality_sweep.py --model models\roberta-base --model_family roberta --dataset sst2 --dataset_dir Codes\framework\data\glue_data\sst2 --output_dir results\unified_sweep\roberta --batch_size 16 --recovery_steps 50 --timeout 1200 --precision fp16 --parent_checkpoint results\roberta_sst2_lora_baseline\finetuned_model.pt --parallel 2 --fresh > logs\roberta_sweep.log 2>&1

echo "Starting GPT-2 Sweep"
python -u Codes\framework\run_quality_sweep.py --model models\gpt2 --model_family gpt --dataset sst2 --dataset_dir Codes\framework\data\glue_data\sst2 --output_dir results\unified_sweep\gpt2 --batch_size 16 --recovery_steps 50 --timeout 1200 --precision fp16 --parent_checkpoint results\gpt2_sst2_lora_baseline\finetuned_model.pt --parallel 2 --fresh > logs\gpt2_sweep.log 2>&1

echo "All sweeps completed"
