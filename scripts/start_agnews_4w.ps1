$env:HF_HOME = "C:\Users\Hadi Schumacher\.cache\huggingface"
$env:PATH = "C:\Users\Hadi Schumacher\anaconda3;C:\Users\Hadi Schumacher\anaconda3\Scripts;C:\Users\Hadi Schumacher\anaconda3\Library\bin;" + $env:PATH
& "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u scripts\run_parallel_dataset_completion.py --dataset agnews --workers 4 > logs\agnews_4w.log 2>&1
