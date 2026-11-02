#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# PBS Torque batch script template for Sharif University HPC Cluster
# Targeting NVIDIA RTX 3090 (24GB VRAM) on GPU-3090-1 node via test-gpu queue
#
# Usage:
#   qsub run.sh
#   qstat -u tahamajlesi.ut
#   qdel <JOB_ID>

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# Job name
#PBS -N neuralops_rtx3090

# Notification: email on abort (a), begin (b), end (e)
#PBS -m abe
#PBS -M tahamajlesi@ut.ac.ir

# Target RTX 3090 (GPU-3090-1 host or test-gpu queue with 1 GPU)
#PBS -l nodes=GPU-3090-1:ppn=2:gpus=1
#PBS -q test-gpu

# Memory request:
#PBS -l mem=16G

# Walltime limit (HH:MM:SS) - test-gpu allows up to 06:00:00:
#PBS -l walltime=04:00:00

# =============================================================================
# JOB EXECUTION
# =============================================================================
set -e

echo "=== Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Queue: $PBS_QUEUE"
echo "Working directory: $PBS_O_WORKDIR"

# Move to submission directory
cd "$PBS_O_WORKDIR"

# Activate NeuralOps conda environment
CENTRAL_ANACONDA="/share/apps/Anaconda/anaconda3.8"
source "$CENTRAL_ANACONDA/bin/activate" "$HOME/.conda/envs/neuralops"

echo "=== Python Environment ==="
echo "Python: $(which python)"
echo "Python Version: $(python --version)"

# Check CUDA and GPU availability
echo "=== GPU Environment Check ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not in PATH; proceeding with PyTorch CUDA check"
fi

echo "=== Strict CUDA verification ==="
python - <<'PYEOF'
import sys
import torch
if torch.version.cuda is None:
    print("FATAL: torch was built without CUDA support (torch.version.cuda is None)")
    sys.exit(2)
print(f"torch.version.cuda = {torch.version.cuda}")
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() is False inside a qsub job")
    sys.exit(3)
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU Allocated: {gpu_name} ({vram_gb:.2f} GB VRAM)")
print("CUDA SUPPORT VERIFIED")
PYEOF

# Execute NeuralOps test suite / benchmark runner
echo "=== Running NeuralOps GPU Verification Suite ==="
python -m pytest tests/conformance/ tests/validator/ -v --tb=short

echo "=== Job Completed Successfully ==="
echo "Date: $(date)"
