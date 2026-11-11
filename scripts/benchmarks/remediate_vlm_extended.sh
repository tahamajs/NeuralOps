#!/bin/bash
# =============================================================================
# Automated Extended Remediation for Low-Accuracy / Low-Delta / Failed VLM Keys
# Uses Offline Teacher Cache (0 MB Teacher VRAM) with Double Training Budget (240s)
# =============================================================================

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/lib:/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/neuralops_run:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"
TEACHER_VLM="/tmp/llava_scienceqa_teacher"
KEYS_JSON="/tmp/neuralops_run/results/bert_attention_zoo/registry_keys.json"
VLM_METRICS="/home/tahamajlesi.ut/NeuralOps/results/vlm_llava_scienceqa_zoo/metrics.jsonl"
VLM_DIR="/home/tahamajlesi.ut/NeuralOps/results/vlm_llava_scienceqa_zoo"

cd /tmp/neuralops_run

echo "=== [VLM EXTENDED REMEDIATION] SCANNING FOR LOW ACCURACY / FAILED KEYS AT $(date) ==="

KEYS_TO_REMEDIATE=$($PYTHON -c "
import json, math, pathlib

metrics_file = pathlib.Path('$VLM_METRICS')
flagged = []
if metrics_file.exists():
    for line in metrics_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = d.get('attn_key')
        st = d.get('status')
        acc_after = float(d.get('acc_after') or 0.0)
        acc_delta = float(d.get('acc_delta') or 0.0)
        ppl_after = d.get('ppl_after')
        
        is_bad = False
        if st != 'PASSED':
            is_bad = True
        elif acc_after < 40.0:
            is_bad = True
        elif acc_delta <= 0.0:
            is_bad = True
        elif ppl_after is None or not math.isfinite(float(ppl_after)) or float(ppl_after) > 10.0:
            is_bad = True
            
        if is_bad and k:
            flagged.append(k)

seen = set()
unique_flagged = [x for x in flagged if not (x in seen or seen.add(x))]
print(json.dumps(unique_flagged))
")

echo "Found $(echo \"$KEYS_TO_REMEDIATE\" | $PYTHON -c 'import json, sys; print(len(json.load(sys.stdin)))') keys needing extended training / remediation:"
echo "$KEYS_TO_REMEDIATE" > /tmp/vlm_extended_retry_keys.json

for KEY in $($PYTHON -c "import json; print('\n'.join(json.load(open('/tmp/vlm_extended_retry_keys.json'))))"); do
    echo ""
    echo "========================================================================"
    echo ">>> [VLM EXTENDED-KD (240s)] Re-running $KEY at $(date) <<<"
    echo "========================================================================"
    
    # Remove previous failed or upload markers to allow clean new run
    rm -f "$VLM_DIR/$KEY/upload_success.json" "$VLM_DIR/$KEY/upload_error.json" 2>/dev/null || true
    rm -f ~/.cache/torch_extensions/*/lock ~/.cache/torch_extensions/*/.ninja_lock 2>/dev/null || true
    
    timeout -s KILL 700 $PYTHON scripts/benchmarks/vlm_llava_scienceqa_zoo.py \
        --model "$TEACHER_VLM" \
        --keys_json "$KEYS_JSON" \
        --attn_key "$KEY" \
        --train_data /home/tahamajlesi.ut/NeuralOps/data_cache/scienceqa_img_train \
        --val_data /home/tahamajlesi.ut/NeuralOps/data_cache/scienceqa_img_val \
        --out_dir "$VLM_DIR" \
        --metrics_log "$VLM_METRICS" \
        --surgery_pattern hybrid_3_1 \
        --train_secs 240.0 \
        --margin_weight 1.0 \
        --batch_size 2 \
        --grad_accum 8 \
        --max_train_samples 800 \
        --max_val_samples 200 \
        --hf_org tahamajs \
        --device cuda
    EC=$?
    if [ $EC -ne 0 ]; then
        echo "{\"attn_key\": \"$KEY\", \"status\": \"FAIL_CRASH\", \"error\": \"exit $EC\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$VLM_METRICS"
    fi
    sleep 2
done

echo "=== [VLM EXTENDED REMEDIATION] FINISHED AT $(date) ==="
