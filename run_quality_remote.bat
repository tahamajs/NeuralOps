@echo off
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=C:\Users\Hadi Schumacher\SVD_Project
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module salt_attention > results\audit\quality_salt.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module xattention > results\audit\quality_xattention.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module nystrom_attention > results\audit\quality_nystrom.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module flash_attention3 > results\audit\quality_flash3.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module gqa_attention > results\audit\quality_gqa.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module lora_attention > results\audit\quality_lora.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module cosformer > results\audit\quality_cosformer.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module performer_attention > results\audit\quality_performer.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module linformer_attention > results\audit\quality_linformer.log 2>&1
start /b python neuralops\benchmarks\run_quality_sweep.py --model gpt2 --model_family gpt2 --module full_attention > results\audit\quality_full.log 2>&1
