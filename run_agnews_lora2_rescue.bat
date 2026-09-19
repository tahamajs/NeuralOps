@echo off
setlocal
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo =====================================================================
echo [TRACK A] AG News REJECTED_EXPANSION Rescue - lora_rank=2
echo =====================================================================

cd /d D:\svd_project

python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_agnews_full ^
  --dataset ag_news ^
  --dataset_dir Codes\framework\data\ag_news ^
  --lora_rank 2 --lora_alpha 4 ^
  --batch_size 8 --parallel 3 ^
  --no_skip ^
  --recovery_steps 100 --finetune_steps 200 ^
  --output_dir results\unified_sweep\distilbert_agnews_full_lora2 ^
  --modules adaptive_span adaptive_sparse adasplash aegis_v1 aegis_v2 asvd bam_umoe block_transformer_hiba chela_slay cluster compressed_attention compressed_mha cosformer custom_attn decay_linear_attn eat expert_choice_moe faq fasa flash3 flash3_fp8 fms full_attention fusion_attn galerkin gated_linear_attn goldfinch griffin_ssd halo_regla hamming hgrn2_outer hla hymba_transx infini jamba jamba_info kda kronecker_tt_lora leka_v1 leka_v2 leka_v3 linformer lora_attn lsh moba ^
  >> logs\distilbert_agnews_lora2_rescue.log 2>&1

echo Done! Results in results\unified_sweep\distilbert_agnews_full_lora2\summary.csv
