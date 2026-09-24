#!/usr/bin/env bash
cd /d/SVD_Project || exit 1
export PYTHONPATH="Codes:."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Job 3: DistilBERT SST-2 Rank Sweep..."

for M in $(cat audit/cat5a_modules.txt); do
    for R in 16 32 64 128 256 384; do
        OUT="results/rank_sweep_sst2/${M}/rank_${R}/"
        # Skip if already completed
        if [ -f "${OUT}/summary.csv" ]; then
            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Skipping ${M} rank=${R} (already done)"
            continue
        fi

        # VRAM safety guard
        while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 4500 ]; do
            sleep 8
        done

        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Running DistilBERT: module=${M}, rank=${R}..."
        python Codes/framework/run_quality_sweep.py \
            --modules "$M" \
            --model models/distilbert_sst2 \
            --model_family bert \
            --dataset sst2 \
            --rank "$R" \
            --recovery_steps 50 \
            --learning_rate 2e-4 \
            --seed 42 \
            --output_dir "$OUT"
    done
done

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished Job 3: DistilBERT SST-2 Rank Sweep."
