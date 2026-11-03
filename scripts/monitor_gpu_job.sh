#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: scripts/monitor_gpu_job.sh JOB_ID [LOG_FILE]" >&2
    exit 2
fi
JOB_ID="$1"
LOG_FILE="${2:-}"
while true; do
    if qstat -f "${JOB_ID}" >/tmp/neuralops_qstat.$$ 2>&1; then
        grep -E "Job Id|job_state|exec_host|resources_used.walltime|exit_status" /tmp/neuralops_qstat.$$ || true
    else
        echo "Job ${JOB_ID} is no longer in qstat."
        break
    fi
    if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
        tail -n 12 "${LOG_FILE}"
    fi
    sleep "${NEURALOPS_MONITOR_INTERVAL:-30}"
done
rm -f /tmp/neuralops_qstat.$$
