import subprocess
import json

modules_order = [
    "fusion_attn", "primal", "gqa", "adaptive_sparse_attention", "adaptive_span",
    "adasplash", "goldfinch_ahn", "halo_regla", "nsa_attention", "nha",
    "squeezing_heads_distiller", "progressive_token_pruning", "eat", "infini_attention",
    "optimized_hybrid_attention", "asvd", "rpc", "svd_mamba", "svd_linear", "res_svd",
    "minimal_aurora", "minimal_nova", "minimal_prism", "minimal_svd", "svd_llm_v2",
    "mla", "dobi", "nystrom", "unified_rope", "hgrn2_gated_deltanet", "hgrn2_layer",
    "soafusedattention", "adaptive_svd_attention", "bam_umoe", "block_transformer_hiba",
    "chela_slay", "cluster", "cosformer", "decay_linear_attention", "expert_choice_moe",
    "flash3", "flash3_fp8", "galerkin_attention", "gated_delta_net", "gated_deltanet_block",
    "griffin_ssd", "hymba_transx", "jamba_info", "vspec", "leka_v1", "leka_v2", "leka_v3",
    "linformer", "tpa", "lora_attention", "xattention", "mamba", "mixattention_moska",
    "moe_deltakv_paged", "nasc_attention", "nexus", "orion", "padre", "performer",
    "phantom", "projection_factorized_attn", "rala", "reap_moe", "regla", "routing",
    "salt", "simple_ssm", "span_expanded", "svd_projection", "template_attention",
    "token_compressed", "moba", "spectra_v2", "compressed_mha", "svd_compressed",
    "spectron", "soa_module", "aegis_v5"
]

powershell_script = """
$modules = @('""" + "','".join(modules_order) + """')
$results = @()

foreach ($m in $modules) {
    $item = [PSCustomObject]@{
        Module = $m
        DistilBERT = "N/A"
        RoBERTa = "N/A"
        GPT2 = "N/A"
    }
    
    # distilbert
    $csv = "D:\\SVD_Project\\results\\unified_$m\\distilbert\\summary.csv"
    if (Test-Path $csv) {
        $lines = Get-Content $csv
        if ($lines.Length -gt 1) {
            $last = $lines[-1] -split ','
            $item.DistilBERT = "$($last[2])% ($($last[1]))"
        }
    }
    
    # roberta
    $csv = "D:\\SVD_Project\\results\\unified_$m\\roberta\\summary.csv"
    if (Test-Path $csv) {
        $lines = Get-Content $csv
        if ($lines.Length -gt 1) {
            $last = $lines[-1] -split ','
            $item.RoBERTa = "$($last[2])% ($($last[1]))"
        }
    }
    
    # gpt2
    $csv = "D:\\SVD_Project\\results\\unified_$m\\gpt2\\summary.csv"
    if (Test-Path $csv) {
        $lines = Get-Content $csv
        if ($lines.Length -gt 1) {
            $last = $lines[-1] -split ','
            $item.GPT2 = "$($last[2])% ($($last[1]))"
        }
    }
    
    $results += $item
}

$results | ConvertTo-Json
"""

with open("/Users/tahamajs/Documents/uni/Project/collect_table.ps1", "w") as f:
    f.write(powershell_script)

