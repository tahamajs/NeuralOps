@echo off
echo Starting Multi-Rank Sweep for targeted zero-run classes...
cd /d "C:\Users\Hadi Schumacher\SVD_Project"
set PYTHONPATH=.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

python neuralops\benchmarks\run_rank_sweeper.py ^
    --modules "svd_linear_attn,svd_compressed_module,rpc_attention_class,regla_attention_class,routing_attention_class,replace_attention_class,nha_attention_class,low_rank_modulated_attention_class,your_custom_attention_class" ^
    --model gpt2 ^
    --model_family gpt ^
    --recovery_steps 500 ^
    --batch_size 8 ^
    --parallel_ranks 2

echo Rank sweep complete!
