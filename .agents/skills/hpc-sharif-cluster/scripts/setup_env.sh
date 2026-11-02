#!/bin/bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# Setup conda environment on Sharif HPC Cluster for NeuralOps
set -e

echo "=== Setting up NeuralOps environment on Sharif HPC ==="

CENTRAL_ANACONDA="/share/apps/Anaconda/anaconda3.8"
ENV_NAME="neuralops"

if [ ! -d "$CENTRAL_ANACONDA" ]; then
    echo "ERROR: Central Anaconda not found at $CENTRAL_ANACONDA"
    echo "Check available Anaconda installations: ls -d /share/apps/Anaconda/*"
    exit 1
fi

echo "Found Central Anaconda at $CENTRAL_ANACONDA"

# Create conda environment if it does not exist
if [ ! -d "$HOME/.conda/envs/$ENV_NAME" ]; then
    echo "Creating conda environment $ENV_NAME (Python 3.10)..."
    "$CENTRAL_ANACONDA/bin/conda" create -p "$HOME/.conda/envs/$ENV_NAME" python=3.10 -c conda-forge --override-channels -y
else
    echo "Environment $ENV_NAME already exists at $HOME/.conda/envs/$ENV_NAME"
fi

# Activate environment
source "$CENTRAL_ANACONDA/bin/activate" "$HOME/.conda/envs/$ENV_NAME"

echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Upgrade pip
pip install --upgrade pip

# Detect CUDA on cluster and pick matching torch wheel
# cu118 / cu121 / cu124 — adjust based on cluster CUDA
# Using Aliyun mirror for stable high-speed transfers without connection drops
TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu121/
echo "Installing CUDA 12.1 enabled PyTorch from $TORCH_INDEX..."
pip install torch==2.4.1+cu121 -f "$TORCH_INDEX" -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 10

# Install dependencies from Tsinghua mirror
pip install pytest transformers datasets huggingface_hub scipy optuna -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install NeuralOps in editable mode
cd "$HOME/NeuralOps"
pip install -e . --no-deps

echo "=== Verifying Installation ==="
python -c "
import sys
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available in build:', torch.version.cuda)
if torch.version.cuda is None:
    print('FATAL: PyTorch was built without CUDA support!')
    sys.exit(1)
print('CUDA build verified!')
"

python -c "import neuralops; print('NeuralOps version:', neuralops.__version__)"

echo "=== Setup Complete! ==="
