Set-Location "D:\SVD_Project"
$pythonExe = "C:\Users\Hadi Schumacher\anaconda3\python.exe"

# 1. Start auto_commit_results.py (syncs to GitHub every 5 minutes)
$gitCommitter = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*auto_commit_results.py 300*" }
if (-not $gitCommitter) {
    Write-Output "Starting auto_commit_results.py in background..."
    Start-Process -FilePath $pythonExe -ArgumentList "auto_commit_results.py 300" -WorkingDirectory "D:\SVD_Project"
} else {
    Write-Output "auto_commit_results.py is already running (PID $($gitCommitter.ProcessId))."
}

# 2. Start auto_upload_hf_results.py (syncs to Hugging Face every 5 minutes)
$hfUploader = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*auto_upload_hf_results.py 300*" }
if (-not $hfUploader) {
    Write-Output "Starting auto_upload_hf_results.py in background..."
    Start-Process -FilePath $pythonExe -ArgumentList "auto_upload_hf_results.py 300" -WorkingDirectory "D:\SVD_Project"
} else {
    Write-Output "auto_upload_hf_results.py is already running (PID $($hfUploader.ProcessId))."
}

# 3. Start benchmark queue runner
$queueRunner = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*run_all_benchmarks_queue.py*" }
if (-not $queueRunner) {
    Write-Output "Starting run_all_benchmarks_queue.py in background..."
    Start-Process -FilePath $pythonExe -ArgumentList "run_all_benchmarks_queue.py --models distilbert,roberta,gpt2 --recovery_steps 50 --skip_existing" -WorkingDirectory "D:\SVD_Project"
} else {
    Write-Output "run_all_benchmarks_queue.py is already running (PID $($queueRunner.ProcessId))."
}
