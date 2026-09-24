@echo off
setlocal
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo =====================================================================
echo [TRACK A] MNLI REJECTED_EXPANSION Rescue - lora_rank=2
echo =====================================================================

cd /d D:\svd_project

python -u Codes\framework\run_quality_sweep.py ^
  --model models\distilbert_mnli_full ^
  --dataset mnli ^
  --dataset_dir Codes\framework\data\glue_data\mnli ^
  --lora_rank 2 --lora_alpha 4 ^
  --batch_size 8 --parallel 3 ^
  --no_skip ^
  --recovery_steps 100 --finetune_steps 200 ^
  --output_dir results\unified_sweep\distilbert_mnli_full_lora2 ^
  --modules adaptive_span adasplash aegis_v1 aegis_v2 asvd block_transformer_hiba chela_slay cluster compressed_attention compressed_mha cosformer decay_linear_attn eat fasa flash3 flash3_fp8 fms full_attention fusion_attn galerkin goldfinch gated_linear_attn griffin_ssd halo_regla hamming hla infini inject_simple_ssm kda kronecker_tt_lora leka_v1 leka_v2 leka_v3 linformer moba moe_deltakv_paged nasc nexus nha nli_softmax nsa nystrom oha padre performer ptp qcqa rala reap_moe regla routing rpc simple_ssm span_expanded spectra_v2 squeezing_heads svd template_attention tlinformer token_compressed unified_rope upi_mamba vig wgqa xattention zeros ^
  >> logs\distilbert_mnli_lora2_rescue.log 2>&1

echo Done! Results in results\unified_sweep\distilbert_mnli_full_lora2\summary.csv
