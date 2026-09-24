# Get compression modules from MNLI summary
$mnli = Import-Csv "D:\svd_project\results\unified_sweep\distilbert_mnli_full\summary.csv"
Write-Host "=== MNLI Status Breakdown ==="
$mnli | Group-Object status | Select-Object Name, Count | Format-Table

Write-Host "=== MNLI Compression Modules (compression_rate > 1.0) ==="
$comp = $mnli | Where-Object { [double]$_.compression_rate -gt 1.0 }
Write-Host ("Count: " + $comp.Count)
$comp | Sort-Object module_name | ForEach-Object { $_.module_name + "`t" + $_.status + "`t" + $_.compression_rate }

Write-Host ""
Write-Host "=== MNLI REJECTED_EXPANSION (for lora_rank=2 rescue) ==="
$rej = $mnli | Where-Object { $_.status -eq "REJECTED_EXPANSION" }
Write-Host ("Count: " + $rej.Count)
$rej | Sort-Object module_name | ForEach-Object { $_.module_name + "`t" + $_.lora_params }

Write-Host ""
Write-Host "=== AG News Status ==="
$ag = Import-Csv "D:\svd_project\results\unified_sweep\distilbert_agnews_full\summary.csv"
Write-Host ("Total AG News rows: " + $ag.Count)
$ag | Group-Object status | Select-Object Name, Count | Format-Table

Write-Host "=== AG News Compression Modules (compression_rate > 1.0) ==="
$ag_comp = $ag | Where-Object { [double]$_.compression_rate -gt 1.0 }
Write-Host ("Count: " + $ag_comp.Count)

Write-Host "=== AG News REJECTED_EXPANSION ==="
$ag_rej = $ag | Where-Object { $_.status -eq "REJECTED_EXPANSION" }
Write-Host ("Count: " + $ag_rej.Count)
$ag_rej | Sort-Object module_name | ForEach-Object { $_.module_name }
