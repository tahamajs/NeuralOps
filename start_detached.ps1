$Action = New-ScheduledTaskAction -Execute "C:\Users\Hadi Schumacher\anaconda3\python.exe" -Argument "-u D:\SVD_Project\finish_benchmark.py" -WorkingDirectory "D:\SVD_Project"
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
Register-ScheduledTask -TaskName "RunBenchmarkQueue" -Action $Action -Trigger $Trigger -User "Hadi Schumacher" -RunLevel Highest -Force
Start-ScheduledTask -TaskName "RunBenchmarkQueue"
