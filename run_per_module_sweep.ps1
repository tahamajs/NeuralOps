Set-Location "D:\SVD_Project"

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HOME = "D:\SVD_Project\.hf_cache"
$env:TRANSFORMERS_CACHE = "D:\SVD_Project\.hf_cache"
$env:HF_DATASETS_CACHE = "D:\SVD_Project\.hf_cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$modules = Get-Content "audit\all_module_names.json" | ConvertFrom-Json
$logFile = "audit\module_test_log.jsonl"


$completed = @{}
if (Test-Path $logFile) {
    Get-Content $logFile | ForEach-Object {
        if ($_ -and $_.Trim()) {
            try {
                $r = $_ | ConvertFrom-Json
                if ($r.module) {
                    $completed[$r.module] = $true
                }
            } catch {}
        }
    }
}

foreach ($mod in $modules) {
    if ($completed.ContainsKey($mod)) {
        Write-Host "SKIP $mod (already done)"
        continue
    }
    $retries = 0
    while ($retries -lt 3) {
        Write-Host "RUN $mod (attempt $($retries+1))"
        python run_one_module.py $mod
        if ($LASTEXITCODE -eq 0) { break }
        $retries++
        Start-Sleep -Seconds 10
    }
    if ($retries -ge 3) {
        Write-Host "FAILED $mod after 3 attempts"
    }
}
Write-Host "ALL MODULES DONE"
