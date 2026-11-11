#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -e

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/lib:/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/tmp/neuralops_run:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python"
cd /tmp/neuralops_run

NFS_L="/home/tahamajlesi.ut/NeuralOps/results/bert_compressor_zoo/multirank_lora_ledger.jsonl"
METRICS_QWEN="/home/tahamajlesi.ut/NeuralOps/results/qwen_attention_zoo/metrics.jsonl"
METRICS_BERT_ATTN="/home/tahamajlesi.ut/NeuralOps/results/bert_attention_zoo/metrics.jsonl"
METRICS_QWEN_COMP="/home/tahamajlesi.ut/NeuralOps/results/qwen_compressor_zoo/metrics.jsonl"
METRICS_VLM="/home/tahamajlesi.ut/NeuralOps/results/vlm_llava_scienceqa_zoo/metrics.jsonl"
METRICS_LLAMA="/home/tahamajlesi.ut/NeuralOps/results/tinyllama_1.1b_zoo/metrics.jsonl"

echo "=== RE-EVALUATING FAILED COMPRESSOR PAIRS (SOLO GPU + 8 CPU THREADS) AT $(date) ==="
FAIL_KEYS=$($PYTHON -c "
import json, pathlib
p = pathlib.Path('$NFS_L')
by_c = {}
if p.exists():
    for l in p.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            by_c.setdefault(d.get('compressor'), []).append(d)
keys = [k for k, recs in sorted(by_c.items()) if k and any(r.get('status') != 'PASSED' for r in recs)]
print(json.dumps(keys))
" 2>/dev/null || echo "[]")

echo "Compressors to retry failed ranks: $FAIL_KEYS"

if [ "$FAIL_KEYS" != "[]" ] && [ -n "$FAIL_KEYS" ]; then
    echo "$FAIL_KEYS" > /tmp/retry_fail_comp_keys.json
    rm -rf /tmp/phase2_retry && mkdir -p /tmp/phase2_retry
    $PYTHON -c "
import json, pathlib
p = pathlib.Path('$NFS_L')
if p.exists():
    passed_lines = [l for l in p.read_text().splitlines() if l.strip() and json.loads(l).get('status') == 'PASSED']
    pathlib.Path('/tmp/phase2_retry/multirank_lora_ledger.jsonl').write_text('\n'.join(passed_lines) + '\n')
" 2>/dev/null || true

    $PYTHON scripts/benchmarks/compressor_multirank_lora_eval.py \
        --keys_json /tmp/retry_fail_comp_keys.json \
        --teacher_path /dev/shm/bert_sst2_teacher \
        --cache_path /tmp/sst2_train_tokens.pt \
        --val_cache_path /tmp/sst2_val_cache.pt \
        --out_dir /tmp/phase2_retry \
        --ranks 256,128,64,32 \
        --device cuda || true

    $PYTHON -c "
import json, pathlib
rows = {}
for src in ['$NFS_L', '/tmp/phase2_retry/multirank_lora_ledger.jsonl']:
    fp = pathlib.Path(src)
    if fp.exists():
        for l in fp.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                pair = (d.get('compressor'), d.get('rank'))
                if pair not in rows or d.get('status') == 'PASSED':
                    rows[pair] = d
out = '\n'.join(json.dumps(v) for v in rows.values()) + '\n'
pathlib.Path('$NFS_L').write_text(out)
print('Final merged compressor rows:', len(rows), 'PASSED:', sum(1 for v in rows.values() if v.get('status') == 'PASSED'))
" 2>/dev/null || true
fi

echo "=== PACKAGING MISSING COMPRESSORS (e.g. modegpt_compressor) AT $(date) ==="
$PYTHON scripts/benchmarks/export_bert_compressor_zoo_hf.py \
    --ledger "$NFS_L" \
    --teacher_path /dev/shm/bert_sst2_teacher \
    --cache_path /tmp/sst2_train_tokens.pt \
    --val_cache_path /tmp/sst2_val_cache.pt \
    --out_dir /home/tahamajlesi.ut/NeuralOps/results/bert_compressor_zoo_hf \
    --device cuda || true

echo "=== FINAL CANONICAL 5-FORMAT EXPORT ACROSS ALL 6 ZOOS AT $(date) ==="
$PYTHON scripts/benchmarks/export_dual_zoo_5formats.py \
    --qwen-ledger "$METRICS_QWEN" \
    --bert-ledger "$NFS_L" \
    --bert-attn-ledger "$METRICS_BERT_ATTN" \
    --qwen-comp-ledger "$METRICS_QWEN_COMP" \
    --vlm-ledger "$METRICS_VLM" \
    --tinyllama-ledger "$METRICS_LLAMA" \
    --output-dir /home/tahamajlesi.ut/NeuralOps/results/canonical_5formats \
    --overwrite \
    --require-parquet || true

echo "=== ALL PIPELINES + 6-ZOO RECONCILIATION COMPLETE AT $(date) ==="
