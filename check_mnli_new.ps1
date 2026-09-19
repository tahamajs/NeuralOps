$csv = "D:\svd_project\results\unified_sweep\distilbert_mnli_full\summary.csv"
$rows = Import-Csv $csv
Write-Host ("Total MNLI rows: " + $rows.Count)
Write-Host ""
Write-Host "=== Status breakdown ==="
$rows | Group-Object status | Sort-Object Count -Descending | Select-Object Name, Count | Format-Table

Write-Host "=== Newly rescued (were REJECTED, now PASS/PASS_LOW) ==="
$rows | Where-Object { $_.status -in @("PASS", "PASS_LOW") -and $_.lora_params -eq "589824" } | 
    Select-Object module_name, status, accuracy_pct, lora_params | Sort-Object module_name | Format-Table

Write-Host "=== MODERATE_EXPANSION modules ==="
$rows | Where-Object { $_.status -eq "MODERATE_EXPANSION" } |
    Select-Object module_name, total_params_before, total_params_after, lora_params, error | Format-Table

Write-Host "=== Still REJECTED_EXPANSION (genuine expansion) ==="
$rows | Where-Object { $_.status -eq "REJECTED_EXPANSION" } |
    Select-Object module_name, total_params_before, total_params_after, lora_params | Format-Table
