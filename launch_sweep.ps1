# Set Environment Variables
$env:PYTHONPATH = "D:\SVD_Project;D:\SVD_Project\Codes"
$env:HF_ENDPOINT = "https://hf.devneeds.ir/"
$env:HF_HOME = "D:\huggingface_cache"
$env:TRANSFORMERS_CACHE = "D:\huggingface_cache"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# 1. Kill any existing python processes
Write-Output "Stopping any running python processes..."
Stop-Process -Name python -Force -ErrorAction SilentlyContinue

# 2. Wait 2 seconds and verify 0 python processes
Start-Sleep -Seconds 2
$pyCount = @(Get-Process python -ErrorAction SilentlyContinue).Count
Write-Output "Remaining python processes: $pyCount"

# 3. Define paths and arguments
$pythonExe = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$scriptPath = "D:\SVD_Project\Codes\framework\run_quality_sweep.py"
$stdoutLog = "D:\SVD_Project\sweep_stdout.log"
$stderrLog = "D:\SVD_Project\sweep_stderr.log"
$pidFile = "D:\SVD_Project\sweep_pid.txt"

# Reset log files
New-Item -ItemType File -Path $stdoutLog -Force | Out-Null
New-Item -ItemType File -Path $stderrLog -Force | Out-Null

$argString = "-u ""$scriptPath"" --modules ""D:\SVD_Project\canonical_modules.txt"" --models distilbert --datasets sst2,mrpc --seeds 42 --epochs 1 --lr 3e-5 --weight_decay 0.01 --batch_size 64 --seq_len 128 --timeout 300 --pretrained --gradient_checkpointing --output_dir ""D:\SVD_Project\results\final_paper_run"""

# 4. Launch background process via Start-Process
Set-Location -Path "D:\SVD_Project"
$proc = Start-Process -FilePath $pythonExe -ArgumentList $argString -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -WindowStyle Hidden -PassThru

# 5. Write PID to marker file
$proc.Id | Out-File -FilePath $pidFile -Encoding ascii -Force
Write-Output "Launched Sweep Coordinator Process ID: $($proc.Id)"
