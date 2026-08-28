#!/usr/bin/env bash
# =============================================================================
# run_massive_sweep.sh
# Massive 10-hour Kaggle GPU sweep — ALL 109 attention types × 3 GLUE tasks
# =============================================================================
# Usage: nohup ./Codes/framework/run_massive_sweep.sh > sweep.log 2>&1 &
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUSH="python3 ${SCRIPT_DIR}/push_to_kaggle_gpu.py"
LOG_FILE="${SCRIPT_DIR}/sweep_master.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Wait for a Kaggle kernel to reach COMPLETE or error ─────────────────────
wait_for_complete() {
    local batch_name="$1"
    log "⏳  Waiting for Kaggle batch '$batch_name' to finish..."
    while true; do
        STATUS=$(python3 "${SCRIPT_DIR}/push_to_kaggle_gpu.py" --status 2>&1)
        log "    Status: $STATUS"
        if echo "$STATUS" | grep -qi "COMPLETE"; then
            log "✅  Batch '$batch_name' COMPLETE"
            $PUSH --fetch
            log "📥  Results fetched for '$batch_name'"
            sleep 5
            return 0
        elif echo "$STATUS" | grep -qi "error\|cancel"; then
            log "⚠️   Batch '$batch_name' ended with ERROR/CANCEL — fetching logs and continuing"
            $PUSH --fetch || true
            sleep 5
            return 0
        fi
        sleep 300  # poll every 5 minutes
    done
}

# ── Dispatch one batch ────────────────────────────────────────────────────────
dispatch_batch() {
    local batch_name="$1"
    local task="$2"
    local attn_types="$3"   # space-separated list
    log "🚀  Dispatching batch '$batch_name'  task=$task"
    log "    Attention types: $attn_types"
    $PUSH --command "python run_kaggle.py sweep --attn_types $attn_types --task $task --epochs 3 --batch_size 32 --seeds 42"
    wait_for_complete "$batch_name"
}

# =============================================================================
# ALL 109 ATTENTION TYPES split into batches of ~25
# =============================================================================
BATCH_A="adaptive_span adaptive_svd adaptive_svd_attn adasplash aegis_hybrid aegis_v2 aegis_v3 aegis_v4 aegis_v5 aegis_v6 asvd bam_umoe block_transformer_hiba chela_slay cluster compressed_attention_layer compressed_attention_layer_energy compressed_mha compression_unified_rope eat expert_choice_moe factorised_matrix_state flash3 flash3_fp8 full"
BATCH_B="fusion_attn galerkin gated_delta_net gated_deltanet_block goldfinch_ahn gqa granite_hybrid griffin_ssd halo_regla hamming hgrn2_gated_deltanet hgrn2_layer hgrn2_outer hla hymba_transx infini jamba2 jamba_info kda leka_v1 leka_v2 leka_v3 linformer lola_attn lora_attn"
BATCH_C="lsh mamba minimal_aurora minimal_nova minimal_prism minimal_svd minimal_svd_v2 mixattention_moska mla moba nexus nha nha_compressor nsa nystrom oha orion orion_module performer phantom phantom_module primal progressive_token_pruning qcqa rala"
BATCH_D="reap_moe regla routing rpc salt_attention samba_m3 samba_moe simple_ssm soa_attn soa_fused_attn soa_fused_module soa_module span_expanded spectra_v2 spectral_mamba spectron squeezing_heads_distiller svd svd_compressed svd_compressed_module svd_linear svd_linear_attn svd_llm_v2 svd_proj_attn token_compressed"
BATCH_E="token_compressed_module tpa upi_stuffed_mamba vig vspec wgqa xattention zeros"

TASKS=("sst2" "mrpc" "qnli")

log "============================================================"
log "  MASSIVE 10-HOUR SWEEP STARTING"
log "  Tasks: ${TASKS[*]}"
log "  Batches per task: 5  (A B C D E)"
log "  Total Kaggle kernel launches: $((${#TASKS[@]} * 5)) = ~$(( ${#TASKS[@]} * 5 )) runs"
log "============================================================"

BATCH_NUM=0
for TASK in "${TASKS[@]}"; do
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  TASK: $TASK"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BATCH_NUM=$((BATCH_NUM + 1)); dispatch_batch "${TASK}_batch_A_${BATCH_NUM}" "$TASK" "$BATCH_A"
    BATCH_NUM=$((BATCH_NUM + 1)); dispatch_batch "${TASK}_batch_B_${BATCH_NUM}" "$TASK" "$BATCH_B"
    BATCH_NUM=$((BATCH_NUM + 1)); dispatch_batch "${TASK}_batch_C_${BATCH_NUM}" "$TASK" "$BATCH_C"
    BATCH_NUM=$((BATCH_NUM + 1)); dispatch_batch "${TASK}_batch_D_${BATCH_NUM}" "$TASK" "$BATCH_D"
    BATCH_NUM=$((BATCH_NUM + 1)); dispatch_batch "${TASK}_batch_E_${BATCH_NUM}" "$TASK" "$BATCH_E"
    log "✅  All batches for task '$TASK' finished."
done

log "============================================================"
log "🏁  MASSIVE SWEEP COMPLETE — ALL TASKS DONE"
log "    Results are in: ${SCRIPT_DIR}/results_kaggle_gpu/"
log "============================================================"
