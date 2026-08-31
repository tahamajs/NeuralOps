@echo off  
cd /d D:\SVD_Project  
"C:\Users\Hadi Schumacher\anaconda3\python.exe" Codes\framework\run_quality_sweep.py --modules modules_worker2.txt --models distilbert,bert,roberta --datasets sst2,mrpc,cola,stsb --results_dir results/worker2_run --epochs 1 --seeds 42 --batch_size 128 --precision fp16 > results\worker2_run\stdout.log 2>&1 
