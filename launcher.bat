cd /d "C:\Users\Hadi Schumacher\SVD_Project"
start /B "" "C:\Users\Hadi Schumacher\SVD_Project\run_full_reuse.bat" > kd_sweep.log 2>&1
start /B "" "C:\Users\Hadi Schumacher\SVD_Project\scripts\run_9_modules_rank_sweep.bat" > rank_sweep.log 2>&1
