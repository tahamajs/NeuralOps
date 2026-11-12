#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# PBS Torque batch submission script for Sharif University HPC Cluster
# Target Queue: N3090 on node GPU-3090-1 (NVIDIA RTX 3090 24GB VRAM)
# Workload: Universal Evaluation & Pre-Benchmark Micro-Training Matrix
#
# Usage:
#   qsub scripts/hpc/qsub_universal_eval_matrix.sh
#   qstat -u "$USER"
#   qdel <JOB_ID>

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# Job name
#PBS -N neuralops_eval_matrix

# Target NVIDIA RTX 3090 node (GPU-3090-1) on N3090 queue
# Request 1 node with 2 CPU cores and 1 GPU
#PBS -l nodes=1:ppn=1
#PBS -q N3090

# Memory allocation (24 GB system RAM)
#PBS -l mem=24G

# Walltime limit (HH:MM:SS) - default 06:00:00 for full matrix screening
#PBS -l walltime=06:00:00

# Stream combination and output paths
#PBS -j oe
#PBS -o neuralops_eval_matrix.o$PBS_JOBID

# Notifications: abort, begin, end
#PBS -m abe
#PBS -M tahamajlesi@ut.ac.ir

# =============================================================================
# ENVIRONMENT & WORKING DIRECTORY
# =============================================================================
set -euo pipefail

echo "=========================================================="
echo "NeuralOps Universal Benchmark Matrix Execution"
echo "Date:              $(date)"
echo "Host:              $(hostname)"
echo "Job ID:            ${PBS_JOBID:-LOCAL}"
echo "Queue:             ${PBS_QUEUE:-N3090}"
echo "Working Directory: ${PBS_O_WORKDIR:-$(pwd)}"
echo "=========================================================="

# 1. Enforce PBS_O_WORKDIR standard per AGENTS.md §3.4
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

export PATH="/share/apps/gcc/gcc-8.2.0/bin:/usr/local/cuda-11.3/bin:${PATH:-}"
export LD_LIBRARY_PATH="/share/apps/gcc/gcc-8.2.0/lib64:/share/apps/gcc/gcc-8.2.0/lib:/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"


# 2. Activate centralized Sharif HPC Conda environment
CENTRAL_ANACONDA="/share/apps/Anaconda/anaconda3.8"
NEURALOPS_ENV="${HOME}/.conda/envs/neuralops"

if [ -f "${CENTRAL_ANACONDA}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${CENTRAL_ANACONDA}/bin/activate" "${NEURALOPS_ENV}"
else
    echo "Warning: ${CENTRAL_ANACONDA} not found, falling back to system conda/python"
fi

# Ensure Python points to neuralops environment
PYTHON_BIN="${NEURALOPS_ENV}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(which python3 || which python)"
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

# 3. Offline Mode Configuration
# Prevents network timeouts on compute nodes lacking outbound access
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# 4. Strict GPU Diagnostics & Verification
echo "=== GPU Environment Telemetry ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not in PATH; validating via PyTorch CUDA interface"
fi

"${PYTHON_BIN}" - <<'PYEOF'
import sys
import torch

if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() is False on compute node. Aborting job.", file=sys.stderr)
    sys.exit(2)

device_name = torch.cuda.get_device_name(0)
device_cap = torch.cuda.get_device_capability(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"CUDA Available: True | Device 0: {device_name}")
print(f"Compute Capability: {device_cap} | VRAM: {total_vram_gb:.2f} GB")
PYEOF

# Process Lock Hygiene: clean orphaned lock files on /dev/shm before starting new training chains
echo "Cleaning orphaned process locks on /dev/shm..."
rm -f /dev/shm/*.lock /dev/shm/*_lock /dev/shm/*.lock.* 2>/dev/null || true

# 5. Output Directory & Run Orchestration
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/hpc_eval_matrix_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Launching Universal Benchmark Evaluation Matrix ==="
echo "Output Directory: ${OUTPUT_DIR}"

# Execute universal evaluation across selected linear attentions & competitor benchmarks
"${PYTHON_BIN}" -m neuralops.cli.bench_cmd eval \
    --model gpt2 \
    --sweep-attentions fms_scan,gla_attention,differential_attention,chunkwise_attention \
    --benchmarks perplexity,flash_attn_tflops,vllm_poisson \
    --pre-train-steps 150 \
    --pre-train-engine multi_level_kd \
    --pre-train-optimizer muon \
    --pre-train-scheduler wsd \
    --auto-calibrate \
    --resume \
    --device cuda:0 \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo "=========================================================="
echo "Execution finished with status code: ${EXIT_CODE}"
echo "Results persisted to: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo "=========================================================="

exit "${EXIT_CODE}"
