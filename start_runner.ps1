Set-Location D:\svd_project
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "D:\svd_project;D:\svd_project\Codes"
Start-Process "C:\Users\Hadi Schumacher\anaconda3\python.exe" -ArgumentList "-u", "D:\svd_project\full_a3_sweep_runner.py" -RedirectStandardOutput "D:\svd_project\results\a3_optimization\sweep_execution_v2.log" -RedirectStandardError "D:\svd_project\results\a3_optimization\sweep_execution_v2.err" -WorkingDirectory "D:\svd_project"
Write-Host "Started full_a3_sweep_runner.py"
