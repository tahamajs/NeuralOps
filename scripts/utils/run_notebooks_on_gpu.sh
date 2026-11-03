#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -e
export PATH="/home/tahamajlesi.ut/bin:/share/apps/gcc/gcc-8.2.0/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/lib:$LD_LIBRARY_PATH"
export LDFLAGS="-B/home/tahamajlesi.ut/bin"
export COMPILER_PATH="/home/tahamajlesi.ut/bin"
export PYTHONPATH="/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH"

cd /home/tahamajlesi.ut/NeuralOps
mkdir -p results

if [ "$(hostname)" != "GPU-3090-1.local" ] && [ "$(hostname)" != "GPU-3090-1" ]; then
    echo "Dispatching execution from $(hostname) to GPU-3090-1..."
    exec ssh GPU-3090-1 "bash /home/tahamajlesi.ut/NeuralOps/scripts/utils/run_notebooks_on_gpu.sh"
fi

echo "======================================================================"
echo "Starting NeuralOps 29 Notebook Suite on $(hostname) at $(date)"
echo "Target Accelerator:"
/usr/local/cuda-11.3/bin/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "GPU node active"
echo "======================================================================"

/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python -u \
    scripts/utils/run_all_notebooks.py \
    --dir notebooks \
    --summary-file results/NOTEBOOKS_EXECUTION_SUMMARY.md

EXIT_CODE=$?
echo "======================================================================"
echo "Finished NeuralOps 29 Notebook Suite at $(date) with exit code ${EXIT_CODE}"
echo "======================================================================"
exit ${EXIT_CODE}
