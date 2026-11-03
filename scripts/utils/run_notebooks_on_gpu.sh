#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -e
export PATH="/home/tahamajlesi.ut/bin:$PATH"
export LDFLAGS="-B/home/tahamajlesi.ut/bin"
export COMPILER_PATH="/home/tahamajlesi.ut/bin"
export PYTHONPATH="/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH"

cd /home/tahamajlesi.ut/NeuralOps
mkdir -p results

echo "======================================================================" >> results/notebooks_execution.log
echo "Starting NeuralOps Master Notebook Suite on RTX 3090 at $(date)" >> results/notebooks_execution.log
echo "======================================================================" >> results/notebooks_execution.log

/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python \
    scripts/utils/run_all_notebooks.py \
    --dir notebooks \
    --summary-file results/NOTEBOOKS_EXECUTION_SUMMARY.md \
    >> results/notebooks_execution.log 2>&1

EXIT_CODE=$?
echo "======================================================================" >> results/notebooks_execution.log
echo "Finished NeuralOps Master Notebook Suite at $(date) with exit code ${EXIT_CODE}" >> results/notebooks_execution.log
echo "======================================================================" >> results/notebooks_execution.log
exit ${EXIT_CODE}
