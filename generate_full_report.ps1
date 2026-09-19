$base = "D:\svd_project\results"

Write-Host "================================================================================"
Write-Host "                COMPREHENSIVE SVD ATTENTION & COMPRESSION REPORT                "
Write-Host "================================================================================"
Write-Host ""

# 1. UNIFIED SWEEPS STATUS
Write-Host "=== 1. UNIFIED BENCHMARK SWEEPS ==="
$sweeps = @(
    @{ Name="DistilBERT MNLI Full"; Path="$base\unified_sweep\distilbert_mnli_full\summary.csv" },
    @{ Name="DistilBERT AG News Full"; Path="$base\unified_sweep\distilbert_agnews_full\summary.csv" },
    @{ Name="DistilBERT SST-2 (base)"; Path="$base\unified_sweep\distilbert\summary.csv" }
)

foreach ($sw in $sweeps) {
    if (Test-Path $sw.Path) {
        $rows = Import-Csv $sw.Path
        Write-Host ("`n[" + $sw.Name + "] Total Modules Evaluated: " + $rows.Count)
        $rows | Group-Object status | Sort-Object Count -Descending | Select-Object Name, Count | Format-Table
        
        $low_acc = $rows | Where-Object { [double]$_.accuracy_pct -lt 50.0 -and $_.status -in @("PASS", "PASS_LOW") }
        if ($low_acc) {
            Write-Host ("  -> Low Accuracy Modules (< 50%):")
            $low_acc | Select-Object module_name, status, accuracy_pct, f1_score, error | Format-Table
        }

        $failures = $rows | Where-Object { $_.status -in @("FAIL", "CRASH", "REJECTED_EXPANSION", "EXTREME_EXPANSION") }
        if ($failures) {
            Write-Host ("  -> Errors & Rejected Modules (Total: " + $failures.Count + "):")
            $failures | Select-Object module_name, status, accuracy_pct, error | Format-Table
        }
    } else {
        Write-Host ($sw.Name + " not found.")
    }
}

# 2. 4-RANK ABLATION RESULTS
Write-Host "`n=== 2. 4-RANK ABLATION RESULTS (MNLI, SST-2) ==="
$ablation_stages = Get-ChildItem "$base\rank_ablation" -Directory | Sort-Object Name

foreach ($st in $ablation_stages) {
    $csv = "$($st.FullName)\summary.csv"
    if (Test-Path $csv) {
        $rows = Import-Csv $csv
        Write-Host ("`n[Ablation Stage: " + $st.Name + "] Total: " + $rows.Count)
        $rows | Group-Object status | Select-Object Name, Count | Format-Table
        
        $low_or_fail = $rows | Where-Object { $_.status -ne "PASS" -or [double]$_.accuracy_pct -lt 50.0 }
        if ($low_or_fail) {
            Write-Host "  -> Low Accuracy or Non-PASS in $($st.Name):"
            $low_or_fail | Select-Object module_name, status, accuracy_pct, compression_rate, error | Format-Table
        }
    }
}
