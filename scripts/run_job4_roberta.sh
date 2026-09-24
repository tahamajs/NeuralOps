#!/usr/bin/env bash
cd /d/SVD_Project || exit 1
export PYTHONPATH="Codes:."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Job 4: RoBERTa SST-2 Rank Sweep..."

for M in $(cat audit/cat5a_modules.txt); do
    for R in 16 32 64 128 256 384; do
        OUT="results/rank_sweep_roberta/${M}/rank_${R}/"
        # Skip if already completed
        if [ -f "${OUT}/summary.csv" ]; then
            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Skipping ${M} rank=${R} (already done)"
            continue
        fi

        # VRAM safety guard
        while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 4500 ]; do
            sleep 8
        done

        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Running RoBERTa: module=${M}, rank=${R}..."
        python Codes/framework/run_quality_sweep.py \
            --modules "$M" \
            --model models/roberta_sst2 \
            --model_family roberta \
            --dataset sst2 \
            --rank "$R" \
            --recovery_steps 50 \
            --learning_rate 5e-5 \
            --seed 42 \
            --output_dir "$OUT"
    done
done

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished Job 4: RoBERTa SST-2 Rank Sweep."
