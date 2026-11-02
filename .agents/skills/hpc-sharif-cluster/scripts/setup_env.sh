#!/bin/bash
# ==============================================================
# NeuralOps one-shot environment bootstrap on Sharif HPC
# Run on cluster: bash setup_env.sh
# ==============================================================
set -euo pipefail

CONDA=/share/apps/Anaconda/anaconda3.8/bin/conda
ENV_NAME=neuralops
REPO_DIR=~/NeuralOps

echo "[1/5] Verify conda..."
"$CONDA" --version

echo "[2/5] Create env (skip if exists)..."
"$CONDA" env list | grep -q "^$ENV_NAME " || "$CONDA" create -n "$ENV_NAME" python=3.10 -y

echo "[3/5] Activate and upgrade pip..."
source /share/apps/Anaconda/anaconda3.8/bin/activate "$ENV_NAME"
pip install --upgrade pip

echo "[4/5] Install deps..."
pip install torch triton pytest transformers datasets huggingface_hub

echo "[5/5] Install NeuralOps (editable)..."
cd "$REPO_DIR"
pip install -e . --no-deps

echo "=== Done ==="
python -c "import neuralops; print('neuralops', neuralops.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
