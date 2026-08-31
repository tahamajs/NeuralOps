
Set-Location D:\SVD_Project
Start-Process -FilePath "C:\Users\Hadi Schumacher\anaconda3\python.exe" -ArgumentList "-u D:\SVD_Project\Codes\framework\run_quality_sweep.py --modules all --models distilbert --datasets sst2 --results_dir results/all_modules_paper_run --epochs 1 --batch_size 64 --timeout 600" -WorkingDirectory "D:\SVD_Project" -RedirectStandardOutput "D:\SVD_Project\results\all_modules_paper_run\sweep_master.log" -RedirectStandardError "D:\SVD_Project\results\all_modules_paper_run\sweep_err.log" -WindowStyle Hidden
