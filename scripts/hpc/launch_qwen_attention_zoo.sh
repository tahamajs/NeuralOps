#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

# Authoritative Qwen Attention Zoo Distillation & Training Runner
# Hardware: GPU-3090-1 (RTX 3090 24GB, CUDA 11.3)
# Base model: /dev/shm/Qwen2.5-0.5B (Resident in RAM disk)

set -e

echo "=== STARTING QWEN ATTENTION ZOO RUNNER ==="
date

export PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/neuralops_run:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CC=/share/apps/gcc/gcc-8.2.0/bin/gcc
export CXX=/share/apps/gcc/gcc-8.2.0/bin/g++
export CUDA_HOME=/usr/local/cuda-11.3

PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"

cd /tmp/neuralops_run

mkdir -p results/qwen_attention_zoo

echo "Running Qwen Attention Zoo with hot-swap..."
$PYTHON scripts/benchmarks/qwen_zoo_train_mix.py \
    --model /dev/shm/Qwen2.5-0.5B \
    --keys_json results/bert_attention_zoo/registry_keys.json \
    --surgery_pattern hybrid_3_1 \
    --seq_len 512 \
    --batch_size 8 \
    --grad_accum 4 \
    --train_secs 300.0 \
    --topk 64 \
    --out_dir /tmp/neuralops_run/results/qwen_attention_zoo \
    --metrics_log /tmp/neuralops_run/results/qwen_attention_zoo/metrics.jsonl \
    --hf_org tahamajs \
    --device cuda

echo "=== QWEN ATTENTION ZOO COMPLETED SUCCESSFULLY ==="
date
