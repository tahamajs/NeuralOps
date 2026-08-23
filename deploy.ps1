Set-Location -Path "D:\SVD_Project"
git pull origin main

Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue

$env:HF_HOME = "D:\huggingface_cache"
$env:TRANSFORMERS_CACHE = "D:\huggingface_cache"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONPATH = "C:\Users\Hadi Schumacher\SVD_Project\Codes;C:\Users\Hadi Schumacher\SVD_Project\Codes\framework"

Set-Location -Path "C:\Users\Hadi Schumacher\SVD_Project\Codes\framework"

Start-Process -FilePath "C:\Users\Hadi Schumacher\anaconda3\python.exe" -ArgumentList "run_suite_multiprocess.py" -WindowStyle Hidden -RedirectStandardOutput "D:\benchmark_run.log" -RedirectStandardError "D:\benchmark_run.log"
Start-Process -FilePath "C:\Users\Hadi Schumacher\anaconda3\python.exe" -ArgumentList "live_monitor.py" -WindowStyle Hidden

