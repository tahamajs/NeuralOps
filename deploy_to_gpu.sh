#!/bin/bash
# Deployment script to push the benchmark code to the remote GPU machine and run it.

REMOTE_HOST="192.168.1.102"
REMOTE_USER="Hadi Schumacher"
SSH_KEY="~/Desktop/host_access"
REMOTE_DIR="~/RL_AEGIS_GPU"

echo "=========================================================="
echo "🚀 Deploying RL_AEGIS Benchmark to GPU Remote Machine"
echo "=========================================================="

# Ensure the remote directory exists
ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# Rsync the codebase to the remote machine
echo "📦 Syncing codebase to remote GPU host..."
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'benchmark_output' -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# Run the benchmark on the remote machine
echo "⚙️ Executing benchmark pipeline on remote GPU..."
ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no "${REMOTE_USER}@${REMOTE_HOST}" << 'REMOTE_SCRIPT'
    cd ~/RL_AEGIS_GPU
    
    echo "Creating virtual environment..."
    python3 -m venv .venv_gpu
    source .venv_gpu/bin/activate
    
    echo "Installing requirements..."
    pip install -U pip
    pip install -r benchmark/requirements.txt
    
    echo "Starting GPU benchmark..."
    # You can change to --models full,svd_linear,performer,etc.
    python3 benchmark/run_benchmark.py --models full,svd_linear
REMOTE_SCRIPT

echo "=========================================================="
echo "✅ Remote GPU execution completed!"
echo "To fetch the results back to your local machine, run:"
echo "rsync -avz -e \"ssh -i ${SSH_KEY}\" \"${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/benchmark_output\" ./"
