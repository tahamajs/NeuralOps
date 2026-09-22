#!/bin/bash

MODEL="models/distilbert_sst2"
FAMILY="bert"
DATASET="sst2"
MODULES="performer_attention cosformer primal_attention galerkin hla_inject"

for RANK in 16 32 64 128 256; do
    echo ""
    echo "=========================================="
    echo "  Running rank=$RANK for 5 modules"
    echo "=========================================="
    
    ssh gpu "powershell -Command \"cd D:\SVD_Project; python Codes/framework/run_quality_sweep.py \
        --modules ${MODULES} \
        --model ${MODEL} \
        --model_family ${FAMILY} \
        --dataset ${DATASET} \
        --rank ${RANK} \
        --learning_rate 5e-5 \
        --recovery_steps 100 \
        --lora_rank 16 \
        --batch_size 8 \
        --parallel 4 \
        --fresh \
        --output_dir results/attention_rank_${RANK}/\""
    
    echo "rank=$RANK done (5 modules)"
done

echo ""
echo "All ranks and modules complete."
