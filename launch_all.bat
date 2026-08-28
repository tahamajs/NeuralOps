@echo off
set PYTHONPATH=D:\SVD_Project;D:\SVD_Project\Codes;%PYTHONPATH%
set HF_ENDPOINT=https://hf.devneeds.ir/
set HF_HOME=D:\huggingface_cache
set TRANSFORMERS_CACHE=D:\huggingface_cache
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d D:\SVD_Project
"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py --modules D:\SVD_Project\canonical_modules.txt --models distilbert --datasets sst2,mrpc --seeds 42 --epochs 1 --lr 3e-5 --weight_decay 0.01 --batch_size 64 --seq_len 128 --timeout 300 --pretrained --gradient_checkpointing --output_dir results/final_paper_run >> final_paper_run.log 2>&1
