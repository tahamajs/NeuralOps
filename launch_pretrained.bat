@echo off
set PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes;%PYTHONPATH%
set HF_ENDPOINT=https://hf.devneeds.ir/
cd /d D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --modules full,svd_linear_attention,minimal_svd,mamba,performer_attention,linformer_attention,aegis_v6,leka_v3,griffin_ssd,samba_m3 --models distilbert --datasets sst2,mrpc --seeds 42,123,777 --epochs 5 --lr 2e-5 --weight_decay 0.01 --batch_size 16 --pretrained --gradient_checkpointing --output_dir results/run_quality_pretrained > run_quality_pretrained.log 2>&1
