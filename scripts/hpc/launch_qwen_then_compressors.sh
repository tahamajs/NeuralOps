#!/bin/bash
# Priority Qwen 2.5 (0.5B) Attention Zoo Distillation & Auto-Resume Compressor Pipeline
# Phase 1: Qwen 2.5 (0.5B) Attention Zoo Distillation (All 110 Modules) on WikiText-2
# Phase 2: Resume remaining Compressors across ranks [256, 128, 64, 32]
# Hardware: GPU-3090-1 (RTX 3090 24GB, CUDA 11.3)

set -e

export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/lib:/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/neuralops_run:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"

cd /tmp/neuralops_run

mkdir -p /tmp/neuralops_run/results/bert_compressor_zoo
mkdir -p /home/tahamajlesi.ut/NeuralOps/results/qwen_attention_zoo

echo "================================================================================"
echo "STARTING PRIORITY QWEN DISTILLATION & COMPRESSION PIPELINE AT $(date)"
echo "================================================================================"

# -----------------------------------------------------------------------------
# PHASE 1 (PRIORITY): Qwen 2.5 (0.5B) Attention Zoo Distillation (All 110 Attentions)
# WikiText-2 PPL with Layer-wise Representation Alignment + Cosine Fine-Tuning
# -----------------------------------------------------------------------------
echo ""
echo ">>> [STAGE 1/2] RUNNING QWEN 2.5 (0.5B) ATTENTION ZOO DISTILLATION (ALL 110 MODULES) <<<"
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
    --out_dir /home/tahamajlesi.ut/NeuralOps/results/qwen_attention_zoo \
    --metrics_log /home/tahamajlesi.ut/NeuralOps/results/qwen_attention_zoo/metrics.jsonl \
    --hf_org tahamajs \
    --device cuda

echo ""
echo ">>> PHASE 1 (QWEN ATTENTION ZOO DISTILLATION) COMPLETE AT $(date) <<<"

# -----------------------------------------------------------------------------
# PHASE 2: 82-Compressor Multi-Rank Pareto & LoRA Recovery Engine (Auto-Resume)
# -----------------------------------------------------------------------------
echo ""
echo ">>> [STAGE 2/2] RESUMING 82-COMPRESSOR MULTI-RANK PARETO & LORA EVALUATION <<<"
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
echo "================================================================================"
echo "ALL DUAL-ZOO PHASES FULLY COMPLETED AT $(date)"
echo "================================================================================"
