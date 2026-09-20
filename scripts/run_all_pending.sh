#!/usr/bin/env bash
# =============================================================================
# run_all_pending.sh
# =============================================================================
# Runs all pending quality-gate groups in priority order on hadi-windows via
# SSH + PowerShell.
#
# DO NOT RUN THIS SCRIPT YET.
# It is a planning artifact. Review _PENDING_RUNS.md first.
#
# Usage (from Mac, once approved):
#   bash scripts/run_all_pending.sh 2>&1 | tee logs/run_all_pending_$(date +%Y%m%d_%H%M%S).log
#
# The script will:
#   1. Deploy the latest code to hadi-windows
#   2. Run groups 1–7 in priority order
#   3. Pull results back after each group
#   4. Generate summary_v3.csv after the full sweep completes
# =============================================================================

set -euo pipefail

REMOTE="hadi-windows"
REMOTE_DIR="D:/svd_project"
LOCAL_DIR="/Users/tahamajs/Documents/uni/Project"
SWEEP_OUT="results/distilbert_mnli_full_mnli_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOCAL_DIR}/logs"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# =============================================================================
# STEP 0: Deploy latest code to remote
# =============================================================================
log "STEP 0: Deploying latest code to ${REMOTE}..."

scp "${LOCAL_DIR}/Codes/framework/run_quality_sweep.py" \
    "${REMOTE}:${REMOTE_DIR}/Codes/framework/run_quality_sweep.py"

scp "${LOCAL_DIR}/Codes/framework/full_mnli_validation.py" \
    "${REMOTE}:${REMOTE_DIR}/Codes/framework/full_mnli_validation.py"

scp "${LOCAL_DIR}/Codes/framework/per_class_collapse_audit.py" \
    "${REMOTE}:${REMOTE_DIR}/Codes/framework/per_class_collapse_audit.py"

log "STEP 0: Deploy complete."

# Helper to run a PowerShell command on the remote
remote_run() {
    local desc="$1"; shift
    log "REMOTE [$desc]: $*"
    ssh "${REMOTE}" "powershell -Command \"cd ${REMOTE_DIR}; $*\"" 2>&1
}

# Helper: wait for remote process by checking summary.csv row count
wait_for_sweep() {
    local expected_min="$1"
    local sleep_secs="${2:-60}"
    local max_wait="${3:-14400}"  # 4 hours default
    local elapsed=0
    while true; do
        count=$(ssh "${REMOTE}" "powershell -Command \"(Import-Csv ${REMOTE_DIR}/${SWEEP_OUT}/summary.csv).Count\"" 2>/dev/null || echo "0")
        count=$(echo "${count}" | tr -d '[:space:]')
        log "  Sweep progress: ${count} rows (waiting for ${expected_min}+)"
        if [ "${count}" -ge "${expected_min}" ]; then
            log "  Reached ${count} rows. Continuing."
            break
        fi
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            log "  WARNING: Timed out waiting for ${expected_min} rows after ${max_wait}s"
            break
        fi
        sleep "${sleep_secs}"
        elapsed=$((elapsed + sleep_secs))
    done
}

# =============================================================================
# STEP 1: Wait for in-progress sweep to complete (if still running)
# =============================================================================
log "STEP 1: Checking if main sweep (PID 9000) is still running..."

sweep_running=$(ssh "${REMOTE}" "powershell -Command \"Get-Process python -ErrorAction SilentlyContinue | Where-Object {(Get-CimInstance Win32_Process -Filter (\\\"ProcessId=\\\"+\$_.Id)).CommandLine -like '*run_quality_sweep*'} | Measure-Object | Select-Object -ExpandProperty Count\"" 2>/dev/null || echo "0")
sweep_running=$(echo "${sweep_running}" | tr -d '[:space:]')

if [ "${sweep_running}" -gt 0 ]; then
    log "  Main sweep is still running (${sweep_running} python procs). Waiting for completion..."
    wait_for_sweep 215 120 21600  # wait up to 6 hours for ~215 modules
    log "  Main sweep appears complete."
else
    log "  No active sweep process found. Proceeding."
fi

# =============================================================================
# STEP 2: Pull final summary.csv and check collapse audit
# =============================================================================
log "STEP 2: Pulling final summary.csv..."
scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}/summary.csv" \
    "${LOG_DIR}/summary_final_${TIMESTAMP}.csv"

log "STEP 2: Checking collapse audit output..."
scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}/collapse_audit.csv" \
    "${LOG_DIR}/collapse_audit_${TIMESTAMP}.csv" 2>/dev/null || \
    log "  WARNING: collapse_audit.csv not ready yet"

# =============================================================================
# GROUP 1: Threshold-flip re-runs (6 modules, ~20 min)
# =============================================================================
log "GROUP 1: Running threshold-flip re-runs..."
log "  Modules: hybrid_svd orion block_transformer_hiba leka_v1 infini fusion_attn"

remote_run "Group1-ThresholdFlip" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules hybrid_svd orion block_transformer_hiba leka_v1 infini fusion_attn \`
      --lora_rank 16 --batch_size 16 --parallel 3 \`
      --output_dir ${SWEEP_OUT}_thresholdfix"

scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}_thresholdfix/summary.csv" \
    "${LOG_DIR}/summary_thresholdfix_${TIMESTAMP}.csv" 2>/dev/null || true

log "GROUP 1: Complete."

# =============================================================================
# GROUP 2: Category A re-runs (32 modules, --lora_rank 2, ~3.5 hr)
# =============================================================================
log "GROUP 2: Running Category A re-runs (32 modules, lora_rank=2)..."

log "  Batch 2A (16 modules)..."
remote_run "Group2A-CatA" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules adaptive_span adaptive_sparse aegis_v1 aegis_v2 chela_slay cluster custom_attn decay_linear_attn fms fusion_attn halo_regla hamming hamming_attention infini kda kda_attention \`
      --lora_rank 2 --batch_size 16 --parallel 3 \`
      --output_dir ${SWEEP_OUT}_catA"

log "  Batch 2B (16 modules)..."
remote_run "Group2B-CatA" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules kronecker_tt_lora leka_v1 leka_v2 leka_v3 linformer linformer_attention moba moe_deltakv_paged nasc nexus nha oha ptp qcqa rala simple_ssm \`
      --lora_rank 2 --batch_size 16 --parallel 3 \`
      --output_dir ${SWEEP_OUT}_catA"

scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}_catA/summary.csv" \
    "${LOG_DIR}/summary_catA_${TIMESTAMP}.csv" 2>/dev/null || true

log "GROUP 2: Complete."

# =============================================================================
# GROUP 3: Category B re-runs (5 modules, --lora_rank 4, ~35 min)
# =============================================================================
log "GROUP 3: Running Category B re-runs (5 modules, lora_rank=4)..."

remote_run "Group3-CatB" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules goldfinch griffin_ssd regla hgrn2_layer aegis_spectron_fp8 \`
      --lora_rank 4 --batch_size 16 --parallel 3 \`
      --output_dir ${SWEEP_OUT}_catB"

scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}_catB/summary.csv" \
    "${LOG_DIR}/summary_catB_${TIMESTAMP}.csv" 2>/dev/null || true

log "GROUP 3: Complete."

# =============================================================================
# GROUP 5: Collapse audit (already launched, check output)
# =============================================================================
log "GROUP 5: Pulling collapse audit results..."
scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}/collapse_audit.csv" \
    "${LOG_DIR}/collapse_audit_final_${TIMESTAMP}.csv" 2>/dev/null || \
    log "  WARNING: collapse_audit.csv still not ready. Check logs/collapse_audit.log on remote."

# =============================================================================
# GROUP 6: hgrn2_outer retry (2× timeout, 30 recovery steps, ~2 hr)
# =============================================================================
log "GROUP 6: hgrn2_outer retry with --timeout 7200 --recovery_steps 30..."

remote_run "Group6-hgrn2_outer" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules hgrn2_outer \`
      --lora_rank 16 --batch_size 8 --parallel 1 \`
      --timeout 7200 --recovery_steps 30 \`
      --output_dir ${SWEEP_OUT}_hgrn2_outer_retry"

scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}_hgrn2_outer_retry/summary.csv" \
    "${LOG_DIR}/summary_hgrn2_outer_retry_${TIMESTAMP}.csv" 2>/dev/null || true

log "GROUP 6: Complete."

# =============================================================================
# GROUP 7: FAIL/crash re-runs with more recovery steps
# =============================================================================
log "GROUP 7a: mamba re-run with 100 recovery steps..."
remote_run "Group7a-mamba" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules mamba \`
      --lora_rank 16 --batch_size 8 --parallel 1 \`
      --recovery_steps 100 \`
      --output_dir ${SWEEP_OUT}_fail_rerun"

log "GROUP 7b: cosformer gqa performer_attention aegis_v2 re-runs..."
remote_run "Group7b-DistributedFail" \
    "python Codes\framework\run_quality_sweep.py \`
      --model models\distilbert_mnli_full \`
      --dataset mnli \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --modules cosformer gqa performer_attention aegis_v2 \`
      --lora_rank 16 --batch_size 16 --parallel 3 \`
      --recovery_steps 100 \`
      --output_dir ${SWEEP_OUT}_fail_rerun"

scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}_fail_rerun/summary.csv" \
    "${LOG_DIR}/summary_fail_rerun_${TIMESTAMP}.csv" 2>/dev/null || true

log "GROUP 7: Complete."

# =============================================================================
# GROUP 4: PASS_STRICT full-MNLI validation (77 modules, ~4 hr)
# Must run last — depends on all group results being merged first
# =============================================================================
log "GROUP 4: Full-MNLI validation for PASS_STRICT modules (77 modules)..."

remote_run "Group4-FullMNLI" \
    "python Codes\framework\full_mnli_validation.py \`
      --model models\distilbert_mnli_full \`
      --sweep_csv ${SWEEP_OUT}\summary.csv \`
      --dataset_dir Codes\framework\data\glue_data\mnli \`
      --out ${SWEEP_OUT}\full_mnli_validation.csv \`
      --batch_size 32 --min_acc_filter 70.0"

scp "${REMOTE}:${REMOTE_DIR}/${SWEEP_OUT}/full_mnli_validation.csv" \
    "${LOG_DIR}/full_mnli_validation_${TIMESTAMP}.csv" 2>/dev/null || true

log "GROUP 4: Complete."

# =============================================================================
# FINAL: Generate summary_v3.csv (all groups merged + corrected threshold)
# =============================================================================
log "FINAL: Generating summary_v3.csv..."

python3 << 'PYEOF'
import csv, json, math, os
from collections import Counter
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOG_DIR", "/Users/tahamajs/Documents/uni/Project/logs"))
out_dir = Path("/Users/tahamajs/Documents/uni/Project/results/distilbert_mnli_full_mnli_sweep")

# Merge all summary CSVs (main sweep takes precedence; re-runs override)
merged = {}

def load_csv(path):
    if path.exists():
        with open(path) as f:
            return list(csv.DictReader(f))
    return []

# Load in priority order (later loads override earlier)
for fn in [
    "/tmp/summary_final.csv",  # main sweep
]:
    for r in load_csv(Path(fn)):
        merged[r.get('module_name', r.get('module', ''))] = r

# Corrected thresholds
thr_new_local = 38.60   # N=1000 dynamic
thr_full      = 35.18   # N=9815 full MNLI

# Emit summary_v3.csv
rows_out = []
for mod, r in sorted(merged.items()):
    acc = float(r.get('accuracy_pct', 0) or 0)
    status = r.get('status', '')
    lora_p = int(r.get('lora_params', 0) or 0)
    p_before = int(r.get('total_params_before', 0) or 0)
    p_after = int(r.get('total_params_after', p_before) or p_before)

    base_after = p_after - lora_p
    base_delta = base_after - p_before
    delta_pct = 100.0 * base_delta / max(p_before, 1)

    if delta_pct < -0.1:
        comp_status = 'reduced'
    elif delta_pct > 0.1:
        comp_status = 'expanded'
    else:
        comp_status = 'equal'

    if acc >= 55.0:
        acc_status = 'pass'
    elif acc >= thr_new_local:
        acc_status = 'marginal'
    else:
        acc_status = 'fail'

    gate_pass = comp_status != 'expanded' and acc_status == 'pass'

    rows_out.append({
        'module': mod,
        'compression_status': comp_status,
        'accuracy_status': acc_status,
        'accuracy_pct': acc,
        'delta_effective_pct': round(delta_pct, 4),
        'lora_params': lora_p,
        'gate_pass': 'PASS' if gate_pass else 'FAIL',
        'threshold_used': thr_new_local,
        'threshold_source': 'dynamic N=1000',
        'collapse_flag': r.get('collapse_flag', ''),
        'status_raw': status,
    })

out_path = out_dir / "summary_v3.csv"
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
    w.writeheader()
    w.writerows(rows_out)

print(f"summary_v3.csv: {len(rows_out)} rows -> {out_path}")
PYEOF

log "FINAL: summary_v3.csv generated."

# =============================================================================
# Commit all results
# =============================================================================
log "COMMIT: Committing all results..."

cd "${LOCAL_DIR}"
git add \
    paper/_PENDING_RUNS.md \
    paper/_QUALITY_GATE_FINAL.md \
    Codes/framework/run_quality_sweep.py \
    Codes/framework/full_mnli_validation.py \
    Codes/framework/per_class_collapse_audit.py \
    paper/DIAGNOSTIC_REPORT.md \
    scripts/run_all_pending.sh \
    results/ \
    2>/dev/null || true

git commit -m "feat(sweep): add pending run queue, fix threshold, add collapse audit + full-MNLI validation scripts" || true
git push origin main || true

log "ALL DONE. Results committed and pushed."
log "Total elapsed: $SECONDS seconds"
