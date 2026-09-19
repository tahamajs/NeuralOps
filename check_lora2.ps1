$f = "D:\svd_project\results\unified_sweep\distilbert_mnli_full_lora2\distilbert_mnli_full_adaptive_span_full_metrics.json"
if (Test-Path $f) {
    $j = Get-Content $f | ConvertFrom-Json
    Write-Host ("Status: " + $j.status)
    Write-Host ("Error: " + $j.error)
    Write-Host ("params_before: " + $j.total_params_before)
    Write-Host ("params_after: " + $j.total_params_after)
    Write-Host ("lora_params: " + $j.lora_params)
    Write-Host ("compression_rate: " + $j.compression_rate)
} else { Write-Host "File not found" }

Write-Host ""
Write-Host "=== MNLI lora2 summary so far ==="
$csv = "D:\svd_project\results\unified_sweep\distilbert_mnli_full_lora2\summary.csv"
if (Test-Path $csv) {
    Import-Csv $csv | Select-Object module_name, status, total_params_before, total_params_after, lora_params, error | Format-Table
}
