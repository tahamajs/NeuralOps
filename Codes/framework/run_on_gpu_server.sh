#!/usr/bin/env bash
# =============================================================================
# NeuralOps: GPU Server Universal Execution & Benchmark Launcher
# =============================================================================
# Usage:
#   bash Codes/framework/run_on_gpu_server.sh --mode diagnostics
#   bash Codes/framework/run_on_gpu_server.sh --mode fast
#   bash Codes/framework/run_on_gpu_server.sh --mode comprehensive
#   bash Codes/framework/run_on_gpu_server.sh --prepare-data
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/Codes:${REPO_ROOT}/Codes/framework:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

MODE="${1:---mode}"
ACTION="${2:-diagnostics}"

if [[ "$MODE" != "--mode" ]]; then
    ACTION="$1"
fi

cd "${REPO_ROOT}"

echo "=============================================================================="
echo "  NEURALOPS GPU SERVER WORKFLOW RUNNER"
echo "  Mode: ${ACTION} | Host: $(hostname) | Directory: ${REPO_ROOT}"
echo "=============================================================================="

case "${ACTION}" in
    diagnostics|test)
        echo "🚀 Running Automated GPU Server Diagnostics & Precision Verification..."
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode diagnostics
        ;;
    prepare-data|data)
        echo "🚀 Preparing & Caching All Benchmark Datasets (GLUE + LM)..."
        python3 "${SCRIPT_DIR}/data/prepare_datasets.py" --all --save_dir "${REPO_ROOT}/prepared_benchmark_data"
        ;;
    fast|benchmark)
        echo "🚀 Executing Fast Multi-Architecture Benchmark Sweep on GPU (BF16)..."
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode fast --precision bf16 --batch_size 32 --output_dir "${REPO_ROOT}/results/gpu_server"
        ;;
    comprehensive|full)
        echo "🚀 Executing Comprehensive Multi-Model Evaluation Matrix on GPU..."
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode comprehensive --precision fp16 --models bert,roberta,t5 --datasets sst2,mrpc --output_dir "${REPO_ROOT}/results/gpu_server"
        ;;
    suite|all)
        echo "🚀 Step 1/3: Running Hardware Diagnostics..."
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode diagnostics
        echo "🚀 Step 2/3: Preparing Datasets..."
        python3 "${SCRIPT_DIR}/data/prepare_datasets.py" --tasks sst2,mrpc,qnli,wikitext2 --save_dir "${REPO_ROOT}/prepared_benchmark_data"
        echo "🚀 Step 3/3: Running Master Benchmark..."
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode fast --precision bf16 --output_dir "${REPO_ROOT}/results/gpu_server"
        ;;
    *)
        echo "Usage: bash Codes/framework/run_on_gpu_server.sh [--mode diagnostics|prepare-data|fast|comprehensive|suite]"
        exit 1
        ;;
esac

echo "=============================================================================="
echo "  GPU SERVER WORKFLOW COMPLETED SUCCESSFULLY!"
echo "  Results stored in: ${REPO_ROOT}/results/gpu_server/"
echo "=============================================================================="
