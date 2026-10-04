@echo off
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=C:\Users\Hadi Schumacher\SVD_Project
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
python scripts\run_kd_all.py --modules salt_attention xattention nystrom_attention flash_attention3 gqa_attention lora_attention cosformer performer_attention linformer_attention full_attention --model_name gpt2 --rank 64 --seq_len 128 --batch_size 2 --recovery_steps 500 --warmup_steps 30 --learning_rate 5e-05 --methods C --device cuda --fp16 --timeout 3600
