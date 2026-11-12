#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

# Master Dual-Zoo Pipeline Runner across ALL Modules
# Phase 1: Qwen 2.5 (0.5B) Attention Zoo Distillation (110 Attention Mechanisms) on WikiText-2
# Phase 2: 82-Compressor Multi-Rank Pareto & LoRA Recovery Engine across ranks [256, 128, 64, 32]
# Hardware: GPU-3090-1 (RTX 3090 24GB, CUDA 11.3)
# All Data & Models: Resident in RAM Disk (/dev/shm) for Zero NFS Latency

set -e

export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/neuralops_run:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"

cd /tmp/neuralops_run

mkdir -p /tmp/neuralops_run/results/bert_compressor_zoo
mkdir -p /tmp/neuralops_run/results/qwen_attention_zoo

echo "================================================================================"
echo "STARTING MASTER DUAL-ZOO PIPELINE AT $(date)"
echo "================================================================================"

# -----------------------------------------------------------------------------
# PHASE 1: 82-Compressor Multi-Rank Pareto & LoRA Recovery Engine (All 82 Compressors)
# Estimated time: ~35 minutes (6.5s per rank x 4 ranks x 82 compressors)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [STAGE 1/2] RUNNING 82-COMPRESSOR MULTI-RANK PARETO & LORA EVALUATION <<<"
date
$PYTHON scripts/benchmarks/compressor_multirank_lora_eval.py \
    --keys_json /tmp/neuralops_run/results/compressor_zoo/registry_keys.json \
    --teacher_path /dev/shm/bert_sst2_teacher \
    --cache_path /tmp/sst2_train_tokens.pt \
    --val_cache_path /tmp/sst2_val_cache.pt \
    --out_dir /tmp/neuralops_run/results/bert_compressor_zoo \
    --ranks "256,128,64,32" \
    --device cuda

echo ""
echo ">>> PHASE 1 (82-COMPRESSOR PARETO & LORA) COMPLETE AT $(date) <<<"

# -----------------------------------------------------------------------------
# PHASE 2: Qwen 2.5 (0.5B) Attention Zoo Distillation (All 110 Attentions)
# WikiText-2 PPL with Stage 1 Subspace Alignment + Stage 2 Micro-chunked LM Head
# -----------------------------------------------------------------------------
echo ""
echo ">>> [STAGE 2/2] RUNNING QWEN 2.5 (0.5B) ATTENTION ZOO DISTILLATION (ALL 110 MODULES) <<<"
date
$PYTHON scripts/benchmarks/qwen_zoo_train_mix.py \
    --model /dev/shm/Qwen2.5-0.5B \
    --keys_json results/bert_attention_zoo/registry_keys.json \
    --surgery_pattern hybrid_3_1 \
    --seq_len 512 \
    --batch_size 8 \
    --grad_accum 4 \
    --train_secs 180.0 \
    --topk 64 \
    --out_dir /tmp/neuralops_run/results/qwen_attention_zoo \
    --metrics_log /tmp/neuralops_run/results/qwen_attention_zoo/metrics.jsonl \
    --hf_org tahamajs \
    --device cuda

echo ""
echo ">>> PHASE 2 (QWEN ATTENTION ZOO) COMPLETE AT $(date) <<<"

# -----------------------------------------------------------------------------
# PHASE 3: Canonical 5-Format Multi-Zoo Persistence Export
# (JSON, JSONL, CSV, Markdown Summary, Parquet)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [STAGE 3/3] EXPORTING DUAL-ZOO CANONICAL 5-FORMAT LEDGERS AT $(date) <<<"
$PYTHON scripts/benchmarks/export_dual_zoo_5formats.py \
    --bert-comp-ledger /tmp/neuralops_run/results/bert_compressor_zoo/multirank_lora_ledger.jsonl \
    --qwen-attn-ledger /tmp/neuralops_run/results/qwen_attention_zoo/metrics.jsonl \
    --output-dir /tmp/neuralops_run/results/dual_zoo_5formats \
    --require-parquet \
    --overwrite

echo ""
echo "================================================================================"
echo "MASTER DUAL-ZOO PIPELINE FULLY COMPLETED AT $(date)"
echo "================================================================================"
