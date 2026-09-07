Set-Location "D:\SVD_Project"
$pythonExe = "C:\Users\Hadi Schumacher\anaconda3\python.exe"

# 1. Schedule Task for Hugging Face Uploader
$actionHF = New-ScheduledTaskAction -Execute $pythonExe -Argument "auto_upload_hf_results.py 300" -WorkingDirectory "D:\SVD_Project"
$triggerHF = New-ScheduledTaskTrigger -Once -At (Get-Date)
Register-ScheduledTask -TaskName "SVD_HF_Uploader" -Action $actionHF -Trigger $triggerHF -Force | Out-Null
Start-ScheduledTask -TaskName "SVD_HF_Uploader"

# 2. Schedule Task for Git Auto-Committer
$actionGit = New-ScheduledTaskAction -Execute $pythonExe -Argument "auto_commit_results.py 300" -WorkingDirectory "D:\SVD_Project"
$triggerGit = New-ScheduledTaskTrigger -Once -At (Get-Date)
Register-ScheduledTask -TaskName "SVD_Git_Committer" -Action $actionGit -Trigger $triggerGit -Force | Out-Null
Start-ScheduledTask -TaskName "SVD_Git_Committer"

# 3. Schedule Task for Benchmark Queue Runner
$actionQueue = New-ScheduledTaskAction -Execute $pythonExe -Argument "run_all_benchmarks_queue.py --models distilbert,roberta,gpt2 --recovery_steps 50 --skip_existing" -WorkingDirectory "D:\SVD_Project"
$triggerQueue = New-ScheduledTaskTrigger -Once -At (Get-Date)
Register-ScheduledTask -TaskName "SVD_Benchmark_Queue" -Action $actionQueue -Trigger $triggerQueue -Force | Out-Null
Start-ScheduledTask -TaskName "SVD_Benchmark_Queue"

Write-Output "All 3 background tasks registered and started in Windows Task Scheduler successfully."
