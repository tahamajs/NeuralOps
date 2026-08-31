@echo off
cd /d "D:\SVD_Project"
set HF_ENDPOINT=https://hf.devneeds.ir/
set PYTHONPATH=D:\SVD_Project\Codes
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --output_dir results/run_quality_fixed_final --modules full,svd_linear_attention,minimal_svd,aegis_v6,leka_v3,nasc_attention,svd_mamba,mamba,samba_m3,gated_deltanet_block,griffin_ssd,jamba2,spectron,block_transformer_hiba,performer_attention,cosformer,linformer_attention,galerkin_attention,adaptive_span,soa_attn --models bert,roberta --datasets sst2,mrpc --seeds 42,123,777 --epochs 3 > run_quality_sweep_final_run.log 2>&1
