#!/usr/bin/env bash
# =============================================================================
# NeuralOps: GPU Server Universal Execution & Benchmark Launcher
# =============================================================================
# Usage:
#   bash Codes/framework/run_on_gpu_server.sh --mode test       # Run hardware diagnostics & module verification
#   bash Codes/framework/run_on_gpu_server.sh --mode benchmark  # Run fast benchmark sweep across attention architectures
#   bash Codes/framework/run_on_gpu_server.sh --mode full       # Run comprehensive multi-dataset evaluation matrix
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/Codes:${REPO_ROOT}/Codes/framework:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

MODE="${1:---mode}"
ACTION="${2:-test}"

if [[ "$MODE" != "--mode" ]]; then
    ACTION="$1"
fi

cd "${REPO_ROOT}"

echo "=============================================================================="
echo "  NEURALOPS GPU SERVER WORKFLOW RUNNER"
echo "  Mode: ${ACTION} | Host: $(hostname) | Directory: ${REPO_ROOT}"
echo "=============================================================================="

case "${ACTION}" in
    test|diagnostics)
        echo "🚀 Running Automated GPU Server Diagnostics & Precision Verification..."
        python3 "${SCRIPT_DIR}/launch_gpu_server_test.py" --output_dir "${REPO_ROOT}/results"
        ;;
    benchmark|fast)
        echo "🚀 Executing Fast Multi-Architecture Benchmark Sweep on GPU..."
        python3 "${REPO_ROOT}/Codes/run_all.py" --mode fast --output_dir "${REPO_ROOT}/results"
        ;;
    full|comprehensive)
        echo "🚀 Executing Full Comprehensive Evaluation Matrix on GPU..."
        python3 "${REPO_ROOT}/Codes/run_all.py" --mode comprehensive --output_dir "${REPO_ROOT}/results"
        ;;
    suite|all)
        echo "🚀 Step 1/2: Running Diagnostics..."
        python3 "${SCRIPT_DIR}/launch_gpu_server_test.py" --output_dir "${REPO_ROOT}/results"
        echo "🚀 Step 2/2: Running Master Benchmark..."
        python3 "${REPO_ROOT}/Codes/run_all.py" --mode fast --output_dir "${REPO_ROOT}/results"
        ;;
    *)
        echo "Usage: bash Codes/framework/run_on_gpu_server.sh [--mode test|benchmark|full|suite]"
        exit 1
        ;;
esac

echo "=============================================================================="
echo "  GPU SERVER WORKFLOW COMPLETED SUCCESSFULLY!"
echo "  Results stored in: ${REPO_ROOT}/results/"
echo "=============================================================================="
