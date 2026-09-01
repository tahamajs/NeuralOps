# Auto-execution script for GPU Server (Windows Powershell)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "D:\SVD_Project;D:\SVD_Project\Codes"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Set-Location "D:\SVD_Project"
Write-Host "[1/4] Pulling latest updates from origin/main..." -ForegroundColor Cyan
git pull origin main

$PythonExe = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "D:\SVD_Project\results\gpu_run_$ts"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "[2/4] Running Qwen & AASVD Master Test Suites on GPU..." -ForegroundColor Cyan
& $PythonExe -m pytest Codes/framework/tests/test_qwen_comprehensive.py Codes/framework/tests/test_aasvd_exhaustive.py Codes/framework/tests/test_model_family_compat.py -v *> "$outDir\qwen_and_svd_tests.log"
Get-Content "$outDir\qwen_and_svd_tests.log" -Tail 15

Write-Host "[3/4] Running Full Master 203 Models Comprehensive Test Suite..." -ForegroundColor Cyan
& $PythonExe Codes/framework/tests/test_all_203_models_comprehensive.py *> "$outDir\master_203_models.log"
Get-Content "$outDir\master_203_models.log" -Tail 15

Write-Host "[4/4] Running Qwen Zero-Shot Evaluation Sweep..." -ForegroundColor Cyan
& $PythonExe -u Codes/framework/run_qwen_zero_shot_sweep.py --output_dir "$outDir\qwen_sweep" *> "$outDir\qwen_sweep.log"

Write-Host "GPU Execution Complete! All results saved to: $outDir" -ForegroundColor Green
