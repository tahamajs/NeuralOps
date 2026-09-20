param(
    [string]$dataset = "mnli",
    [string]$model = "models/distilbert_mnli_full",
    [string]$datasetDir = "Codes/framework/data/glue_data/mnli"
)

$modules = "a3 aasvd adaptive adaptive_svd adasvd aegis_hybrid aegis_v3 aegis_v4 aegis_v5 aegis_v6 come compressed_attention_layer_energy cur_moe cur_moe_class fused_svd_kernel gqa hybrid_svd inject_aegis_hybrid inject_aegis_v6 inject_minimal_aurora inject_minimal_svd inject_minimal_svd_v2 inject_svd_proj_attn inject_swift_svd kmeans kronecker lola minimal_aurora minimal_nova minimal_prism minimal_svd minimal_svd_v2 mla orion phantom projection_factorized_attn replaceme res_svd ressvd salt soa_attn soa_fused_attention spectral_residual_lora spectron svd_compressed svd_linear svd_linear_decay svd_llm_v2 svd_proj swift_svd swift_svd_class vspec"

$modList = $modules -split " "

$argList = @(
    "-u", "Codes\framework\run_quality_sweep.py",
    "--model", $model,
    "--dataset", $dataset,
    "--dataset_dir", $datasetDir,
    "--rank", "192",
    "--recovery_steps", "50",
    "--lora_rank", "16",
    "--output_dir", "results\rank_ablation\$($dataset)_rank192",
    "--modules"
) + $modList

Start-Process python -ArgumentList $argList -WorkingDirectory "D:\svd_project" -RedirectStandardOutput "logs\ablation_$($dataset)_r192.log" -RedirectStandardError "logs\ablation_$($dataset)_r192_err.log"
Write-Output "Launched rank 192 sweep for $dataset"
