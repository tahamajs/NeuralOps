cd /d D:\SVD_Project
start "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" scripts\run_monitor_server.py > results\mon.log 2>&1
start "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" auto_upload_hf_results.py 180 > results\up.log 2>&1
start "" "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u run_all_benchmarks_queue.py > results\master_q.log 2>&1
