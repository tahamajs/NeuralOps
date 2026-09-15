@echo off
cd /d D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u "D:\SVD_Project\Codes\runners\run_all_benchmarks_queue.py" --repo_dir "D:\SVD_Project\Codes\runners" > "D:\SVD_Project\results\full_benchmark\queue_stdout.log" 2> "D:\SVD_Project\results\full_benchmark\queue_stderr.log"
