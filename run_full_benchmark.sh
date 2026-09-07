#!/usr/bin/env bash
# NeuralOps container verification suite (Dockerfile CMD).
#
# Runs the same minimal reproducibility sweep the CI workflow runs, so
# `docker run neuralops:2026` performs a real end-to-end forward/fine-tune/
# evaluation pass instead of exiting silently. Override with arguments to run a
# bigger sweep: `docker run neuralops:2026 bash run_full_benchmark.sh <extra args>`
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/Codes"

SMOKE_DIR="${SMOKE_DIR:-/tmp/neuralops_smoke}"

echo "== NeuralOps smoke benchmark =="
python Codes/framework/run_quality_sweep.py \
  --max_steps 1 \
  --seed 42 \
  --modules full \
  --model bert-base-uncased \
  --model_family bert \
  --dataset sst2 \
  --batch_size 16 \
  --output_dir "${SMOKE_DIR}" \
  "$@"

test -f "${SMOKE_DIR}/summary.csv" || { echo "FAIL: summary.csv missing"; exit 1; }
test -f "${SMOKE_DIR}/detailed_results.json" || { echo "FAIL: detailed_results.json missing"; exit 1; }

echo "== Smoke benchmark PASS =="
cat "${SMOKE_DIR}/summary.csv"
