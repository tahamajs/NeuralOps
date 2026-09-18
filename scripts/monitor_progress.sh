#!/bin/bash
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "===== Progress @ $TS ====="

# AG News progress
AG_TOTAL=$(wc -l < audit/agnews_completion_2026_09_18/remaining.txt 2>/dev/null || echo 0)
AG_DONE=$(grep -c '"status": "OK"' audit/agnews_completion_2026_09_18/checkpoint.jsonl 2>/dev/null || echo 0)
AG_ERR=$(grep -c '"status": "ERROR"' audit/agnews_completion_2026_09_18/checkpoint.jsonl 2>/dev/null || echo 0)
echo "AG News: $AG_DONE / $((AG_TOTAL * 3)) (errors: $AG_ERR)"

# MNLI progress
MN_TOTAL=$(wc -l < audit/mnli_completion_2026_09_18/remaining.txt 2>/dev/null || echo 0)
MN_DONE=$(grep -c '"status": "OK"' audit/mnli_completion_2026_09_18/checkpoint.jsonl 2>/dev/null || echo 0)
MN_ERR=$(grep -c '"status": "ERROR"' audit/mnli_completion_2026_09_18/checkpoint.jsonl 2>/dev/null || echo 0)
echo "MNLI:    $MN_DONE / $((MN_TOTAL * 3)) (errors: $MN_ERR)"

# GPU status
ssh gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null

# Last 3 completed
echo "Last completed AG News:"
tail -n 2 audit/agnews_completion_2026_09_18/checkpoint.jsonl 2>/dev/null
echo "Last completed MNLI:"
tail -n 2 audit/mnli_completion_2026_09_18/checkpoint.jsonl 2>/dev/null

echo ""
