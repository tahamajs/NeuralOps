Set-Location 'D:\SVD_Project'
$mods = @(
    'gqa_attention',
    'gqa',
    'compressed_mha',
    'compressed_mha_attention',
    'fused_svd_kernel',
    'adaptive_svd_attention',
    'aegis_hybrid',
    'aegis_hybrid_attention',
    'lora_attention',
    'lola_attention',
    'lsh_attention'
)

$outPath = 'D:\SVD_Project\audit\category_a_results.jsonl'
New-Item -ItemType File -Force -Path $outPath | Out-Null

foreach ($m in $mods) {
    Write-Host "Running: $m"
    $json = & python run_one_module.py $m --gpt2-only --no-log
    Add-Content -Path $outPath -Value $json
}
Write-Host "Category A batch complete."
