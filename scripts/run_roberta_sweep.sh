#!/bin/bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd ~/Project
source .venv/bin/activate

python -m neuralops.benchmarks.cli run \
    --model roberta-base \
    --dataset sst2 \
    --all-attn \
    --device cuda \
    --epochs 1 \
    --batch-size 16 \
    --lr 2e-5 \
    --max-samples 1500 \
    --max-eval-samples 300 \
    --timeout-s 600 \
    2>&1 | tee ~/roberta_benchmark_fast.log
