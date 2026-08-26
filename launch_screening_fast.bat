@echo off
set PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes;%PYTHONPATH%
set HF_ENDPOINT=https://hf.devneeds.ir/
cd /d D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --modules full,svd_linear_attention,minimal_svd,aegis_v6,leka_v3,nasc_attention,svd_mamba,mamba,samba_m3,gated_deltanet_block,griffin_ssd,jamba2,spectron,block_transformer_hiba,performer_attention,cosformer,linformer_attention,galerkin_attention,adaptive_span,soa_attn --models distilbert --datasets sst2 --seeds 42 --epochs 2 --lr 2e-5 --weight_decay 0.01 --batch_size 128 --pretrained --gradient_checkpointing --output_dir results/run_quality_screening_fast > run_quality_screening_fast.log 2>&1
