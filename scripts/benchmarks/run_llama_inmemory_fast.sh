#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

# Fast unified single-process in-memory runner for TinyLlama Attention Zoo.
# Loads base model ONCE to CPU RAM, then swaps attention in-memory for each key.
set -euo pipefail

export PYTHONPATH="/home/tahamajlesi.ut/NeuralOps:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export PYTHONUNBUFFERED="1"

PYTHON="/home/tahamajlesi.ut/miniconda3/envs/neuralops/bin/python"
WORKDIR="/home/tahamajlesi.ut/NeuralOps"
cd "$WORKDIR"

echo "=== Launching In-Memory Fast TinyLlama-1.1B Attention Zoo ==="
exec "$PYTHON" scripts/benchmarks/llama_zoo_train_mix.py \
    --model /home/tahamajlesi.ut/NeuralOps/models/tinyllama_1.1b \
    --keys_json results/bert_attention_zoo/registry_keys.json \
    --surgery_pattern hybrid_3_1 \
    --seq_len 512 \
    --batch_size 4 \
    --grad_accum 8 \
    --train_secs 90.0 \
    --out_dir /home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo \
    --metrics_log /home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo/metrics.jsonl \
    --hf_org tahamajs \
    --device cuda
