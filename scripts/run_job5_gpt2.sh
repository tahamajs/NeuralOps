#!/usr/bin/env bash
cd /d/SVD_Project || exit 1
export PYTHONPATH="Codes:."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Job 5: GPT-2 Wikitext-2 Rank Sweep..."

for M in a3_compressor svd_llm_compressor saes_svd_compressor slicegpt swift_svd_act tt_compressor; do
    for R in 16 32 64 128 256; do
        OUT="results/rank_sweep_gpt2/${M}/rank_${R}/"
        # Skip if already completed
        if [ -f "${OUT}/summary.csv" ]; then
            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Skipping ${M} rank=${R} (already done)"
            continue
        fi

        # VRAM safety guard
        while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 4500 ]; do
            sleep 8
        done

        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Running GPT-2: module=${M}, rank=${R}..."
        python Codes/framework/run_quality_sweep.py \
            --modules "$M" \
            --model models/gpt2 \
            --model_family gpt2 \
            --dataset wikitext2 \
            --rank "$R" \
            --recovery_steps 50 \
            --learning_rate 5e-5 \
            --seed 42 \
            --output_dir "$OUT"
    done
done

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished Job 5: GPT-2 Wikitext-2 Rank Sweep."
