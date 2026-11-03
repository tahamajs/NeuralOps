#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUEUE="N3090"
WALLTIME="24:00:00"
JOB_NAME="neuralops_gpu"
COMMAND="scripts/run_gpu.sh validate-cuda"
DETACHED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue) QUEUE="$2"; shift 2 ;;
        --walltime) WALLTIME="$2"; shift 2 ;;
        --name) JOB_NAME="$2"; shift 2 ;;
        --command) COMMAND="$2"; shift 2 ;;
        --screen) DETACHED=1; shift ;;
        -h|--help) echo "Usage: scripts/submit_gpu.sh [--queue N3090] [--walltime 24:00:00] [--name NAME] [--command '...'] [--screen]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
JOB_FILE="$(mktemp "${TMPDIR:-/tmp}/neuralops_gpu.XXXXXX.pbs")"
trap 'rm -f "${JOB_FILE}"' EXIT
cat > "${JOB_FILE}" <<EOF
#!/usr/bin/env bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l nodes=1:ppn=1
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o ${ROOT_DIR}/results/logs/${JOB_NAME}.\${PBS_JOBID}.log
set -euo pipefail
cd ${ROOT_DIR}
export PYTHONPATH="${ROOT_DIR}:\${PYTHONPATH:-}"
export HF_HUB_OFFLINE="\${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="\${TRANSFORMERS_OFFLINE:-1}"
${COMMAND}
EOF
if (( DETACHED )); then
    command -v screen >/dev/null || { echo "screen is required for --screen" >&2; exit 1; }
    SESSION="${JOB_NAME}_$(date +%s)"
    screen -dmS "${SESSION}" bash -lc "qsub '${JOB_FILE}'"
    echo "Submitted through detached screen session: ${SESSION}"
else
    qsub "${JOB_FILE}"
fi
