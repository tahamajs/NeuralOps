$ErrorActionPreference = 'Stop'
Set-Location 'D:\SVD_PROJECT\Codes\framework'
$env:HF_HOME = 'C:\Users\Hadi Schumacher\.cache\huggingface'
$env:TRANSFORMERS_CACHE = 'C:\Users\Hadi Schumacher\.cache\huggingface\hub'
$env:PYTHONPATH = 'D:\SVD_PROJECT\Codes;D:\SVD_PROJECT\Codes\framework'
& 'C:\Users\Hadi Schumacher\anaconda3\python.exe' -u 'D:\SVD_PROJECT\Codes\framework\run_quality_sweep.py' `
  --model 'Qwen/Qwen2.5-0.5B-Instruct' `
  --dataset_dir 'D:\SVD_PROJECT\Codes\framework\data\glue_data\sst2' `
  --dataset sst2 --modules all --max_steps 1 --finetune_steps 200 `
  --seq_len 64 --timeout 1800 --precision fp16 --batch_size 8 `
  --gradient_checkpointing_use_reentrant False `
  --output_dir 'D:\SVD_PROJECT\results\qwen_full_classification'
