#!/usr/bin/env bash
cd /d/SVD_Project || exit 1
export PYTHONPATH="Codes:."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Job 2: Linformer batch=16 Benchmark..."

# Wait if VRAM is heavily utilized
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 5000 ]; do
    sleep 5
done

python Codes/framework/run_seqlen_scaling_bench.py \
    --modules linformer_attention \
    --model models/gpt2 \
    --seq_lens 128,256,512,1024 \
    --batch_size 16 \
    --output_dir results/linformer_b16/

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished Job 2: Linformer Benchmark."
