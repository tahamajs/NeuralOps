$env:HF_HOME = "C:\Users\Hadi Schumacher\.cache\huggingface"
$env:PATH = "C:\Users\Hadi Schumacher\anaconda3;C:\Users\Hadi Schumacher\anaconda3\Scripts;C:\Users\Hadi Schumacher\anaconda3\Library\bin;" + $env:PATH
& "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --modules a3 --model models\roberta-base --dataset sst2 --recovery_steps 0 --batch_size 2 --output_dir results\a3_tests\roberta_zeroshot\ > logs\roberta_zeroshot_a3.log 2>&1
