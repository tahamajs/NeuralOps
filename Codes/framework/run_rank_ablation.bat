@echo off
setlocal
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /d D:\svd_project

REM =====================================================================
REM TRACK B: 4-Rank Ablation & Publication Pipeline for Compressors
REM Ranks: 32, 64, 128, 256 (4 ranks) across MNLI, SST-2, AG News
REM After completion, automatically aggregates tables and uploads to Hugging Face
REM =====================================================================

set COMP_MODULES=a3 aasvd adaptive adaptive_svd adasvd aegis_hybrid aegis_v3 aegis_v4 aegis_v5 aegis_v6 come compressed_attention_layer_energy cur_moe cur_moe_class fused_svd_kernel gqa hybrid_svd inject_aegis_hybrid inject_aegis_v6 inject_mamba inject_minimal_aurora inject_minimal_svd inject_minimal_svd_v2 inject_svd_proj_attn inject_swift_svd kmeans kronecker kronecker_linear_class lola mamba minimal_aurora minimal_nova minimal_prism minimal_svd minimal_svd_v2 mla orion phantom projection_factorized_attn replaceme res_svd ressvd salt soa_attn soa_fused_attention spectral_residual_lora spectron svd_compressed svd_linear svd_linear_decay svd_llm_v2 svd_proj swift_svd swift_svd_class vspec

echo =====================================================================
echo [RANK ABLATION] Stage 1/3: MNLI - Ranks: 32 64 128 256
echo =====================================================================

for %%R in (32 64 128 256) do (
  echo --- MNLI rank=%%R ---
  python -u Codes\framework\run_quality_sweep.py ^
    --model models\distilbert_mnli_full ^
    --dataset mnli ^
    --dataset_dir Codes\framework\data\glue_data\mnli ^
    --rank %%R ^
    --batch_size 8 --parallel 3 --no_skip ^
    --recovery_steps 80 --finetune_steps 150 ^
    --output_dir results\rank_ablation\mnli_rank%%R ^
    --modules %COMP_MODULES% ^
    >> logs\rank_ablation_mnli_rank%%R.log 2>&1
  echo Finished MNLI rank=%%R
)

echo =====================================================================
echo [RANK ABLATION] Stage 2/3: SST-2 - Ranks: 32 64 128 256
echo =====================================================================

for %%R in (32 64 128 256) do (
  echo --- SST-2 rank=%%R ---
  python -u Codes\framework\run_quality_sweep.py ^
    --model models\distilbert_sst2 ^
    --dataset sst2 ^
    --dataset_dir Codes\framework\data\glue_data\sst2 ^
    --rank %%R ^
    --batch_size 8 --parallel 3 --no_skip ^
    --recovery_steps 80 --finetune_steps 150 ^
    --output_dir results\rank_ablation\sst2_rank%%R ^
    --modules %COMP_MODULES% ^
    >> logs\rank_ablation_sst2_rank%%R.log 2>&1
  echo Finished SST-2 rank=%%R
)

echo =====================================================================
echo [RANK ABLATION] Stage 3/3: AG News - Ranks: 32 64 128 256
echo =====================================================================

for %%R in (32 64 128 256) do (
  echo --- AG News rank=%%R ---
  python -u Codes\framework\run_quality_sweep.py ^
    --model models\distilbert_agnews_full ^
    --dataset ag_news ^
    --dataset_dir Codes\framework\data\ag_news ^
    --rank %%R ^
    --batch_size 8 --parallel 3 --no_skip ^
    --recovery_steps 80 --finetune_steps 150 ^
    --output_dir results\rank_ablation\agnews_rank%%R ^
    --modules %COMP_MODULES% ^
    >> logs\rank_ablation_agnews_rank%%R.log 2>&1
  echo Finished AG News rank=%%R
)

echo =====================================================================
echo [POST-PROCESSING] Aggregating results & Generating Pareto Frontiers
echo =====================================================================
python Codes\framework\aggregate_rank_ablation.py --results_dir results\rank_ablation >> logs\rank_ablation_aggregation.log 2>&1

echo =====================================================================
echo [HUGGING FACE] Publishing PASS models and metrics to Hugging Face
echo =====================================================================
python Codes\framework\upload_ablation_to_hf.py --base_dir results\rank_ablation --ranks 32 64 128 256 >> logs\rank_ablation_hf_upload.log 2>&1

echo =====================================================================
echo ALL 4-RANK ABLATIONS AND HF UPLOADS COMPLETED SUCCESSFULLY!
echo =====================================================================
