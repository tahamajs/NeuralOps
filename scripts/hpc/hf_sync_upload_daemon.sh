#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

# Asynchronous Background Sync and Hugging Face Upload Daemon
# Runs on HPC login node (with outbound internet access)
# Monitors both bert_attention_zoo and qwen_attention_zoo

export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_ENABLE_HF_TRANSFER=0
PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"
LOCAL_RESULTS="/home/tahamajlesi.ut/NeuralOps/results"
mkdir -p "$LOCAL_RESULTS/bert_attention_zoo" "$LOCAL_RESULTS/qwen_attention_zoo"

echo "=== STARTING UNIFIED HF SYNC & UPLOAD DAEMON ON LOGIN NODE ==="
date

while true; do
    # 1. Sync completed models from GPU node for BERT
    rsync -aq --include="*/" --include="*.safetensors" --include="*.json" --include="*.py" --include="*.md" --exclude="*" \
        GPU-3090-1:/tmp/neuralops_run/results/bert_attention_zoo/ "$LOCAL_RESULTS/bert_attention_zoo/" 2>/dev/null

    # 2. Sync completed models from GPU node for Qwen
    rsync -aq --include="*/" --include="*.safetensors" --include="*.json" --include="*.py" --include="*.md" --exclude="*" \
        GPU-3090-1:/tmp/neuralops_run/results/qwen_attention_zoo/ "$LOCAL_RESULTS/qwen_attention_zoo/" 2>/dev/null

    # 3. Check for completed un-uploaded BERT models
    for dir in "$LOCAL_RESULTS/bert_attention_zoo"/*; do
        if [ -d "$dir" ] && { [ -f "$dir/attention_weights.safetensors" ] || [ -f "$dir/model.safetensors" ] || [ -f "$dir/pytorch_model.bin" ]; } && [ ! -f "$dir/upload_success.json" ]; then
            key=$(basename "$dir")
            echo "[UPLOAD DAEMON] Uploading BERT model: $key to Hugging Face..."
            $PYTHON /home/tahamajlesi.ut/NeuralOps/scripts/hpc/bert_zoo_upload_hf.py \
                --single_dir "$dir" \
                --hf_org "Hooshaai" \
                --cleanup 1 >> /tmp/hf_uploader_daemon.log 2>&1 || true
        fi
    done

    # 4. Check for completed un-uploaded Qwen models
    for dir in "$LOCAL_RESULTS/qwen_attention_zoo"/*; do
        if [ -d "$dir" ] && { [ -f "$dir/model.safetensors" ] || [ -f "$dir/attention_weights.safetensors" ] || [ -f "$dir/pytorch_model.bin" ]; } && [ ! -f "$dir/upload_success.json" ]; then
            key=$(basename "$dir")
            echo "[UPLOAD DAEMON] Uploading Qwen model: $key to Hugging Face..."
            $PYTHON /home/tahamajlesi.ut/NeuralOps/scripts/hpc/bert_zoo_upload_hf.py \
                --single_dir "$dir" \
                --hf_org "Hooshaai" \
                --cleanup 1 >> /tmp/hf_uploader_daemon.log 2>&1 || true
        fi
    done

    sleep 15
done
