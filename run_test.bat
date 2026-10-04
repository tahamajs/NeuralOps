@echo off
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
python scripts\run_kd_reuse_model.py --modules xattention salt_attention nystrom_attention --recovery_steps 100 --batch_size 8
