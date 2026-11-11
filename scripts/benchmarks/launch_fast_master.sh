#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -e

# Kill any lingering background tasks
pkill -f chain_llama_vlm 2>/dev/null || true
pkill -f llama_zoo_train_mix 2>/dev/null || true
pkill -f post_chain_reconcile 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/lib:/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/neuralops_run:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"
cd /tmp/neuralops_run

echo "=== [$(date)] MASTER FAST PIPELINE STARTED ==="
echo "Python: $($PYTHON --version)"
echo "Torch: $($PYTHON -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))')"

echo "=== [1/2] RUNNING FAST IN-MEMORY TINYLLAMA ZOO ==="
$PYTHON scripts/benchmarks/llama_zoo_train_mix.py \
    --model /home/tahamajlesi.ut/NeuralOps/models/tinyllama_1.1b \
    --keys_json /tmp/neuralops_run/results/bert_attention_zoo/registry_keys.json \
    --surgery_pattern hybrid_3_1 \
    --seq_len 512 \
    --batch_size 4 \
    --grad_accum 8 \
    --train_secs 90.0 \
    --out_dir /home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo \
    --metrics_log /home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo/metrics.jsonl \
    --hf_org tahamajs \
    --device cuda

echo "=== [2/2] RUNNING POST-CHAIN RECONCILIATION & 5-FORMAT EXPORT ==="
bash /tmp/post_chain_reconcile_solo.sh

echo "=== [$(date)] ALL 6 ZOOS & 5 FORMATS 100% COMPLETE ==="
