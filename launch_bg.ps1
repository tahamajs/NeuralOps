param (
    [string]$ModulesFile = "D:\SVD_Project\canonical_modules.txt",
    [string]$OutputDir = "D:\SVD_Project\results\final_paper_run"
)

# 1. Kill any existing python processes
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 2. Environment
$env:PYTHONPATH = "D:\SVD_Project;D:\SVD_Project\Codes"
$env:HF_ENDPOINT = "https://hf.devneeds.ir/"
$env:HF_HOME = "D:\huggingface_cache"
$env:TRANSFORMERS_CACHE = "D:\huggingface_cache"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# 3. Create cmd launcher batch file
$batContent = @"
@echo off
set PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes;%PYTHONPATH%
set HF_ENDPOINT=https://hf.devneeds.ir/
set HF_HOME=D:\huggingface_cache
set TRANSFORMERS_CACHE=D:\huggingface_cache
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --modules "$ModulesFile" --models distilbert --datasets sst2,mrpc --seeds 42 --epochs 1 --lr 3e-5 --weight_decay 0.01 --batch_size 64 --seq_len 128 --timeout 300 --pretrained --gradient_checkpointing --output_dir "$OutputDir" > D:\SVD_Project\sweep_stdout.log 2>&1
"@

$batContent | Out-File -FilePath "D:\SVD_Project\run_sweep_worker.bat" -Encoding ascii -Force

# 4. Launch detached cmd process
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "D:\SVD_Project\run_sweep_worker.bat" -WindowStyle Hidden -PassThru
$proc.Id | Out-File -FilePath "D:\SVD_Project\sweep_pid.txt" -Encoding ascii -Force
Write-Output "Launched worker batch process with PID: $($proc.Id)"
