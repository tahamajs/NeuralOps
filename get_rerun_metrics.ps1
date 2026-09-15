
$rerun_mods = @('gqa','adaptive_sparse_attention','squeezing_heads_distiller','svd_mamba','svd_linear','minimal_nova','minimal_prism','svd_llm_v2','mla','dobi','unified_rope','hgrn2_gated_deltanet','soafusedattention','gated_delta_net','gated_deltanet_block','vspec','leka_v1','tpa','mamba','orion','padre','performer','phantom','projection_factorized_attn','reap_moe','regla','routing','salt','simple_ssm','svd_projection','template_attention','token_compressed','compressed_mha','svd_compressed','spectron','soa_module')

$res = @()
foreach ($m in $rerun_mods) {
    $item = [PSCustomObject]@{
        Module = $m
        DistilBERT = "N/A"
        RoBERTa = "N/A"
        GPT2 = "N/A"
        LastTime = ""
    }
    
    foreach ($backbone in @('distilbert', 'roberta', 'gpt2')) {
        $p = "D:\SVD_Project\results\unified_$m\$backbone\summary.csv"
        if (Test-Path $p) {
            $lines = Get-Content $p
            if ($lines.Length -gt 1) {
                $last = $lines[-1] -split ','
                $item.$backbone = "$($last[2])% ($($last[1]))"
                $item.LastTime = (Get-Item $p).LastWriteTime.ToString("HH:mm:ss")
            }
        }
    }
    $res += $item
}
$res | ConvertTo-Json
