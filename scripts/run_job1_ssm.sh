#!/usr/bin/env bash
cd /d/SVD_Project || exit 1
export PYTHONPATH="Codes:."

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Job 1: SSM Scaling Benchmark..."
python Codes/framework/run_seqlen_scaling_bench.py \
    --modules mamba,mamba_svd_flash,samba_m3,gated_deltanet_block,hgrn2_layer,hgrn2_outer,simple_ssm \
    --model models/gpt2 \
    --seq_lens 128,256,512,1024 \
    --batch_size 16 \
    --output_dir results/seqlen_groupC/

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished Job 1: SSM Scaling Benchmark."
