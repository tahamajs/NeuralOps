#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -u
cd /home/tahamajlesi.ut/NeuralOps

export PY="/home/tahamajlesi.ut/NeuralOps/scripts/run_py_gpu.sh"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PWD}"

echo "START=$(date -Iseconds)"
bash "$PY" - <<'PY'
import sys, torch
print("TORCH=", torch.__version__)
print("CUDA=", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    sys.exit(1)
PY

FAILURES=0
PASSES=0
for script in $(ls examples/[0-9]*.py | sort); do
    base=$(basename "$script")
    echo "=================================================="
    echo "=== BEGIN $base $(date -Iseconds) ==="
    echo "=================================================="
    bash "$PY" "$script"
    rc=$?
    echo "=== END $base rc=$rc $(date -Iseconds) ==="
    if [ $rc -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
        echo "❌ FAILED: $base (exit code $rc)"
    else
        PASSES=$((PASSES + 1))
        echo "✅ PASSED: $base"
    fi
done

echo "=================================================="
echo "RUN COMPLETED: $(date -Iseconds) | PASSED: $PASSES | FAILED: $FAILURES"
echo "=================================================="
