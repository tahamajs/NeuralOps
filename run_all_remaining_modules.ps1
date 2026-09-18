$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_ORG = "Hooshaai"
$env:PYTHONWARNINGS = "ignore"
$env:OMP_NUM_THREADS = "8"

Set-Location D:\svd_project

Write-Host "=== Starting Full Remaining Module Pipeline with Hooshaai & Batch Size 128 ===" -ForegroundColor Green

# 1. Primary Aegis Target Suite
$aegis_tasks = @(
    @{mod="aegis_v2"; model="models/distilbert_sst2"; family="bert"; ds="sst2"},
    @{mod="aegis_v2"; model="models/roberta-base"; family="roberta"; ds="mnli"},
    @{mod="aegis_v3"; model="models/gpt2"; family="gpt2"; ds="agnews"},
    @{mod="aegis_v4"; model="models/gpt2"; family="gpt2"; ds="agnews"},
    @{mod="aegis_v5"; model="models/distilbert_sst2"; family="bert"; ds="sst2"}
)

foreach ($t in $aegis_tasks) {
    Write-Host ">>> Executing $($t.mod) on $($t.ds) ($($t.model))" -ForegroundColor Cyan
    python -u Codes/framework/run_quality_sweep.py `
        --modules $t.mod `
        --model $t.model `
        --model_family $t.family `
        --dataset $t.ds `
        --rank 64 `
        --batch_size 128 `
        --recovery_steps 50 `
        --lora_rank 16 `
        --precision fp16 `
        --hf_org "Hooshaai" `
        --upload_to_hf `
        --output_dir "results/fixed_aegis/$($t.mod)_$($t.ds)/"
    Start-Sleep -Seconds 5
}

# 2. Comprehensive Queue for Remaining Checked Modules
$remaining_modules = @(
    "adaptive_span_attention", "adaptive_sparse_attention", "adaptive_svd_attention", "adasplash_attention",
    "aegis_hybrid", "aegis_v1", "asvd_attention", "chela_slay_attention", "cluster_attention",
    "cosformer_attention", "decay_gated_linear_attention", "decay_linear_attn", "expert_choice_moe_attention",
    "factorised_matrix_state", "fasa_attention", "flash_attention3", "full_attention", "galerkin_attention",
    "gated_linear_attn", "gqa_attention", "granite_hybrid_block", "griffin_ssd_block", "halo_regla_attention",
    "hamming_attention", "hgrn2", "hla_attention", "hymba_transx", "infini_attention", "jamba2",
    "kda", "leka_v1", "leka_v2_attention", "leka_v3_attention", "lora_attention", "low_rank_modulated_attention",
    "lsh_attention", "mamba", "mamba_svd_flash", "minimal_aurora", "minimal_nova", "minimal_prism",
    "minimal_svd", "minimal_svd_v2", "mixattention_moska", "mla_attention", "moe_deltakv_paged",
    "nasc_attention", "nexus", "nha_attention", "nli_softmax", "nsa_attention", "nystrom_attention",
    "optimized_hybrid_attention", "orion_attention", "padre_attention", "performer_attention", "primal_attention",
    "projection_factorized_attn", "qcqa_attention", "rala_linear_attention", "reap_moe", "replace_attention",
    "res_svd_attention", "routing_attention", "rpc_attention", "salt", "samba_m3", "samba_moe",
    "simple_ssm", "smart_weighted_ensemble", "soa_fused_attention", "span_expanded_attention",
    "spectraformer_attention", "spectron_attn", "squeezing_heads_module", "svd_compressed_module",
    "svd_linear", "svd_llm_v2_attention", "svd_mamba", "svd_sequence_attention", "tlinformer_attention",
    "token_compressed_module", "tp_attention", "upi_mamba", "vig_attention", "wgqa_attention", "zeros_attention"
)

Write-Host ">>> Starting queue of all remaining verified & patched attention modules" -ForegroundColor Yellow
foreach ($m in $remaining_modules) {
    Write-Host ">>> Queue processing module: $m on distilbert_sst2" -ForegroundColor Cyan
    try {
        python -u Codes/framework/run_quality_sweep.py `
            --modules $m `
            --model models/distilbert_sst2 `
            --model_family bert `
            --dataset sst2 `
            --rank 64 `
            --batch_size 128 `
            --recovery_steps 50 `
            --lora_rank 16 `
            --precision fp16 `
            --hf_org "Hooshaai" `
            --upload_to_hf `
            --output_dir "results/all_modules/$m/"
    } catch {
        Write-Host "Module $m failed or threw error: $_" -ForegroundColor Red
    }
    Start-Sleep -Seconds 3
}

Write-Host "=== Entire Module Queue Completed ===" -ForegroundColor Green
