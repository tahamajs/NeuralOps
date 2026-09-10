$Action1 = New-ScheduledTaskAction -Execute "C:\Users\Hadi Schumacher\anaconda3\python.exe" -Argument "-u D:\SVD_Project\auto_upload_hf_results.py 180" -WorkingDirectory "D:\SVD_Project"
$Trigger1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
Register-ScheduledTask -TaskName "RunUploader" -Action $Action1 -Trigger $Trigger1 -User "Hadi Schumacher" -RunLevel Highest -Force
Start-ScheduledTask -TaskName "RunUploader"

$Action2 = New-ScheduledTaskAction -Execute "C:\Users\Hadi Schumacher\anaconda3\python.exe" -Argument "-u D:\SVD_Project\auto_git_push.py" -WorkingDirectory "D:\SVD_Project"
$Trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
Register-ScheduledTask -TaskName "RunGitPush" -Action $Action2 -Trigger $Trigger2 -User "Hadi Schumacher" -RunLevel Highest -Force
Start-ScheduledTask -TaskName "RunGitPush"
