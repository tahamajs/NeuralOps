Set-Location D:\svd_project
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "D:\svd_project;D:\svd_project\Codes"
python -u D:\svd_project\full_a3_sweep_runner.py >> D:\svd_project\results\a3_optimization\sweep_execution_v2.log 2>&1
