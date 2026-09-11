# 1. SSH config for GitHub port 443
$sshConfigDir = "$env:USERPROFILE\.ssh"
$sshConfigFile = "$sshConfigDir\config"
if (-not (Test-Path $sshConfigDir)) { New-Item -ItemType Directory -Path $sshConfigDir | Out-Null }
$sshEntry = @"

Host github.com
  HostName ssh.github.com
  Port 443
  User git
"@
$existing = if (Test-Path $sshConfigFile) { Get-Content $sshConfigFile -Raw } else { "" }
if (-not ($existing -match "ssh.github.com")) {
    Add-Content -Path $sshConfigFile -Value $sshEntry
    Write-Host "[OK] SSH config updated for port 443"
} else {
    Write-Host "[SKIP] SSH config already has port 443"
}

# 2. Kill old queue script
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like '*run_all_benchmarks_queue*' -or $_.CommandLine -like '*finish_benchmark*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "[OK] Old queue processes killed"

# 3. Pull latest code
Set-Location D:\SVD_Project
git pull origin main -X theirs
Write-Host "[OK] Git pull done"

# 4. Register Scheduled Tasks with AtStartup trigger
$python = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$svdDir = "D:\SVD_Project"

# Task: finish_benchmark
$action_fb = New-ScheduledTaskAction -Execute $python -Argument "-u $svdDir\finish_benchmark.py" -WorkingDirectory $svdDir
$trigger_startup = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 24)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "SVD_FinishBenchmark" -Action $action_fb -Trigger $trigger_startup -Settings $settings -Principal $principal -Force
Write-Host "[OK] SVD_FinishBenchmark task registered"

# Task: uploader
$action_up = New-ScheduledTaskAction -Execute $python -Argument "-u $svdDir\auto_upload_hf_results.py 180" -WorkingDirectory $svdDir
Register-ScheduledTask -TaskName "SVD_Uploader" -Action $action_up -Trigger $trigger_startup -Settings $settings -Principal $principal -Force
Write-Host "[OK] SVD_Uploader task registered"

# 5. Launch finish_benchmark in background (survives SSH disconnect)
$job = Start-Job -ScriptBlock {
    Set-Location D:\SVD_Project
    & "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u D:\SVD_Project\finish_benchmark.py 2>&1 | Tee-Object -FilePath D:\SVD_Project\finish_benchmark.log -Append
}
Write-Host "[OK] finish_benchmark.py launched as background job (ID: $($job.Id))"
Write-Host ""
Write-Host "All infrastructure setup complete!"
