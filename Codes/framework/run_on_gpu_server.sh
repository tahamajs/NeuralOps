#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/Codes:${REPO_ROOT}/Codes/framework:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Parse: support both "bash run.sh fast" and "bash run.sh --mode fast"
ACTION="${1:-diagnostics}"
[[ "$ACTION" == "--mode" ]] && ACTION="${2:-diagnostics}"

cd "${REPO_ROOT}"
echo "============================================================"
echo " NeuralOps GPU Runner | Mode: ${ACTION} | Host: $(hostname)"
echo "============================================================"

case "${ACTION}" in
    diagnostics|test)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode diagnostics ;;
    prepare-data|data)
        python3 "${SCRIPT_DIR}/data/prepare_datasets.py" \
            --all --save_dir "${REPO_ROOT}/prepared_benchmark_data" ;;
    fast|benchmark)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" \
            --mode fast --precision bf16 --batch_size 16 \
            --output_dir "${REPO_ROOT}/results/gpu_server" ;;
    comprehensive|full)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" \
            --mode comprehensive --precision fp16 \
            --output_dir "${REPO_ROOT}/results/gpu_server" ;;
    suite|all)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode diagnostics
        python3 "${SCRIPT_DIR}/data/prepare_datasets.py" \
            --tasks sst2,mrpc,qnli --save_dir "${REPO_ROOT}/prepared_benchmark_data"
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" \
            --mode fast --precision bf16 \
            --output_dir "${REPO_ROOT}/results/gpu_server" ;;
    report)
        python3 "${SCRIPT_DIR}/auto_run.py" --report ;;
    *)
        echo "Usage: bash run_on_gpu_server.sh [diagnostics|prepare-data|fast|comprehensive|suite|report]"
        exit 1 ;;
esac
echo "============================================================"
echo " Completed! Results in: ${REPO_ROOT}/results/gpu_server/"
echo "============================================================"
