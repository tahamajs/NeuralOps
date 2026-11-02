#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# PBS Torque batch script template for Sharif University HPC Cluster
#
# Usage:
#   qsub run.sh
#   qstat -u tahamajlesi.ut
#   qdel <JOB_ID>

# =============================================================================
# PBS DIRECTIVES
# =============================================================================
# Job name
#PBS -N neuralops_gpu_job

# Notification: email on abort (a), begin (b), end (e)
#PBS -m abe
#PBS -M tahamajlesi@ut.ac.ir

# Resource requests:
# nodes=1:ppn=2:gpus=1 -> 1 node, 2 CPU cores, 1 GPU
#PBS -l nodes=1:ppn=2:gpus=1

# Memory request:
#PBS -l mem=16G

# Walltime limit (HH:MM:SS):
#PBS -l walltime=02:00:00

# Target Queue (adjust based on cluster status, e.g. cuda9, cuda7, DP, DDP):
#PBS -q cuda9

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
    print("Reinstall with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(2)
print(f"torch.version.cuda = {torch.version.cuda}")
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() is False inside a qsub job")
    print("This means the job was not allocated a GPU, or the driver is broken.")
    sys.exit(3)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("CUDA SUPPORT VERIFIED")
PYEOF

# Execute NeuralOps test suite / benchmark runner
echo "=== Running NeuralOps GPU Verification Suite ==="
python -m pytest tests/conformance/ tests/validator/ -v --tb=short

echo "=== Job Completed Successfully ==="
echo "Date: $(date)"
