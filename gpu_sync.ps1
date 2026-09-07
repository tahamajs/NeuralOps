# gpu_sync.ps1 — Run this on the GPU server (D:\svd_project) to pull all fixes from main
# Usage: powershell -ExecutionPolicy Bypass -File gpu_sync.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  GPU SERVER SYNC — pulling from origin/main" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Set-Location D:\svd_project

# Pull latest main
git fetch origin
git checkout main
git pull origin main

Write-Host ""
Write-Host "=== Latest commits ===" -ForegroundColor Green
git log --oneline -10

Write-Host ""
Write-Host "=== Verifying key fixes ===" -ForegroundColor Green

# 1. Check runner files are self-contained (no framework.runners import)
$broken = Get-ChildItem run_all_4_*.py | Select-String "framework.runners" | Select-Object -ExpandProperty Filename | Sort-Object -Unique
if ($broken) {
    Write-Host "[FAIL] Still broken runners: $($broken -join ', ')" -ForegroundColor Red
} else {
    Write-Host "[OK] All run_all_4_*.py runners are self-contained" -ForegroundColor Green
}

# 2. Check compat.py has the output_gate fix
$compatFix = Select-String "output_gate" Codes\framework\utils\compat.py
if ($compatFix) {
    Write-Host "[OK] compat.py has output_gate trainability fix" -ForegroundColor Green
} else {
    Write-Host "[FAIL] compat.py is missing output_gate fix" -ForegroundColor Red
}

# 3. Check run_quality_sweep.py has kvtc args
$kvtcFix = Select-String "kvtc_bits" Codes\framework\run_quality_sweep.py
if ($kvtcFix) {
    Write-Host "[OK] run_quality_sweep.py has --kvtc_* CLI args" -ForegroundColor Green
} else {
    Write-Host "[FAIL] run_quality_sweep.py missing --kvtc_* args" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  SYNC COMPLETE — GPU server is up to date" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  python run_all_4_leka_v1.py" -ForegroundColor White
Write-Host "  python run_all_4_kvtc.py" -ForegroundColor White
Write-Host "  python run_all_4_linformer.py" -ForegroundColor White
