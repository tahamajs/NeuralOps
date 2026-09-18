/**
 * physics/token_trajectory.js
 * ============================
 * Token 3D Trajectory & Mathematical Physics Engine.
 * 
 * Maps vertical token flow progress (y) into 3D positions (x, z) reflecting
 * the model's actual computational graph and tensor connectivity:
 *   - DistilBERT / RoBERTa: SVD low-rank bottleneck (compression into rank-64 manifold)
 *   - GPT-2: Fused c_attn Conv1D projection
 *   - Qwen2.5/3.5: GQA multi-query routing (14Q : 2KV) followed by SwiGLU MLP
 *   - MoE Paradigms: Dynamic router dispatch to 4 parallel expert blocks
 *   - Residual Skip Highways: 25% of tokens carry the identity signal directly into (+).
 */

import { VISUAL_CONFIG } from '../constants.js';

const STRIDE = VISUAL_CONFIG.TOKENS.layerStride; // 3.2 units

/**
 * Computes a token's exact (x, z) coordinates and active sublayer designation
 * based on its vertical position along the model height.
 * 
 * @param {object} t - Token state { isResidual, lateralNoise, subStream, swigluStream, expertTarget, headIdx }
 * @param {number} y - Current vertical scene height in Three.js units
 * @param {boolean} isCompressed - True for right column (compressed), False for left column (baseline)
 * @param {string} modelKey - 'distilbert' | 'roberta' | 'gpt2' | 'qwen'
 * @param {string} paradigm - 'svd' | 'moe' | 'mla' | 'slicegpt' | 'quant'
 * @param {number} baseOffsetX - Column center X position (-4.5 or +4.5)
 * @returns {{ x: number, z: number, sublayer: string }} 3D position and semantic sublayer tag
 */
export function computeToken3DTrajectory(t, y, isCompressed, modelKey, paradigm, baseOffsetX) {
    // 1. Stage 00: Input Embeddings (Base Plate)
    if (y < 1.0) {
        return {
            x: baseOffsetX + t.lateralNoise * 0.4,
            z: t.lateralNoise * 0.25,
            sublayer: 'input_embeddings'
        };
    }

    // 2. Intra-layer normalized progress: tau in [0.0, 1.0)
    const layerProgress = y - 1.0;
    const currentLayerIdx = Math.floor(layerProgress / STRIDE);
    const tau = (layerProgress % STRIDE) / STRIDE;

    // ── RESIDUAL SKIP HIGHWAY DYNAMICS ─────────────────────────
    // 25% of tokens bypass sublayers to visualize the identity skip connection:
    // x = x + Attention(Norm(x)) and x = x + FFN(Norm(x))
    if (t.isResidual) {
        // Highway bypass lateral conduit offset
        const conduitX = baseOffsetX + (isCompressed ? 1.85 : -1.85);

        // Dip 1: Merging into Attention Residual Add (+) at tau ~ 0.42 - 0.48
        if (tau >= 0.40 && tau <= 0.48) {
            const dipAlpha = Math.sin(((tau - 0.40) / 0.08) * Math.PI);
            const xPos = conduitX + (baseOffsetX - conduitX) * dipAlpha;
            return { x: xPos, z: 0, sublayer: 'residual_add_attn' };
        }

        // Dip 2: Merging into FFN Residual Add (+) at tau ~ 0.90 - 0.98
        if (tau >= 0.88 && tau <= 0.98) {
            const dipAlpha = Math.sin(((tau - 0.88) / 0.10) * Math.PI);
            const xPos = conduitX + (baseOffsetX - conduitX) * dipAlpha;
            return { x: xPos, z: 0, sublayer: 'residual_add_ffn' };
        }

        // Standard bypass travel
        return {
            x: conduitX + t.lateralNoise * 0.15,
            z: 0.1,
            sublayer: 'residual'
        };
    }

    // ── COMPUTATIONAL CORE (Main Tensor Pathway) ───────────────

    // Sublayer Phase A: Input LayerNorm (tau: 0.00 - 0.12)
    if (tau < 0.12) {
        return {
            x: baseOffsetX + t.lateralNoise * 0.25,
            z: 0,
            sublayer: 'pre_attn_norm'
        };
    }

    // Sublayer Phase B: Self-Attention Sublayer (tau: 0.12 - 0.46)
    if (tau < 0.46) {
        // ARCHITECTURE 1: Qwen GQA Multi-Query Projection (14Q : 2KV)
        if (modelKey === 'qwen') {
            if (t.subStream === 'q') {
                // Query branch (wide fan-out)
                const qOffset = -0.70 + (t.headIdx / 14.0) * 0.40;
                return { x: baseOffsetX + qOffset, z: -0.1, sublayer: 'q_proj' };
            } else if (t.subStream === 'k') {
                // Key branch (compact 2-head bundle)
                return { x: baseOffsetX + 0.10, z: 0.1, sublayer: 'k_proj' };
            } else {
                // Value branch (compact 2-head bundle)
                return { x: baseOffsetX + 0.65, z: 0.1, sublayer: 'v_proj' };
            }
        }

        // ARCHITECTURE 2: GPT-2 Fused c_attn Conv1D Projection
        if (modelKey === 'gpt2') {
            // All tokens travel together through the fused QKV projection slab
            return {
                x: baseOffsetX + t.lateralNoise * 0.55,
                z: 0,
                sublayer: isCompressed ? 'v_factor' : 'c_attn'
            };
        }

        // ARCHITECTURE 3: SVD Low-Rank Bottleneck (DistilBERT / RoBERTa)
        if (isCompressed && paradigm === 'svd') {
            if (tau < 0.28) {
                // Projection V factor (sharp pinch into rank-64 bottleneck)
                return { x: baseOffsetX + t.lateralNoise * 0.18, z: 0, sublayer: 'v_factor' };
            } else {
                // Reconstruction U factor (expansion back to hidden dimension)
                return { x: baseOffsetX + t.lateralNoise * 0.60, z: 0, sublayer: 'u_factor' };
            }
        }

        // Standard Dense Baseline Attention (Q, K, V split)
        const qkvOffset = (t.subStream === 'q') ? -0.55 : (t.subStream === 'k' ? 0.0 : 0.55);
        return {
            x: baseOffsetX + qkvOffset + t.lateralNoise * 0.15,
            z: 0,
            sublayer: 'attention_core'
        };
    }

    // Sublayer Phase C: Attention Residual Merge (+) (tau: 0.46 - 0.52)
    if (tau < 0.52) {
        return {
            x: baseOffsetX,
            z: 0,
            sublayer: 'attn_add_node'
        };
    }

    // Sublayer Phase D: Post-Attn Norm & FFN Sublayer (tau: 0.52 - 0.88)
    if (tau < 0.88) {
        // PARADIGM: Mixture of Experts (MoE) 4-Way Routing
        if (isCompressed && paradigm === 'moe') {
            if (tau < 0.60) {
                // MoE Gating Router
                return { x: baseOffsetX, z: 0, sublayer: 'moe_router' };
            } else {
                // Dispatched to specific routed expert [0..3]
                const expertSpacing = 0.85;
                const expertX = baseOffsetX + (t.expertTarget - 1.5) * expertSpacing;
                return { x: expertX, z: 0, sublayer: `expert_${t.expertTarget}` };
            }
        }

        // ARCHITECTURE: Qwen SwiGLU Gated Feed-Forward Network
        if (modelKey === 'qwen') {
            if (t.swigluStream === 'gate') {
                // Gate branch (SiLU activation)
                return { x: baseOffsetX - 0.55, z: 0, sublayer: 'gate_proj' };
            } else {
                // Up projection branch
                return { x: baseOffsetX + 0.55, z: 0, sublayer: 'up_proj' };
            }
        }

        // DistilBERT / RoBERTa / GPT-2 MLP (Intermediate 4x expansion -> Down projection)
        const expansionX = (tau < 0.70) ? (t.lateralNoise * 0.90) : (t.lateralNoise * 0.35);
        return {
            x: baseOffsetX + expansionX,
            z: 0,
            sublayer: (tau < 0.70) ? 'ffn_up' : 'ffn_down'
        };
    }

    // Sublayer Phase E: Layer Exit (+) Residual Addition (tau: 0.88 - 1.00)
    return {
        x: baseOffsetX + t.lateralNoise * 0.20,
        z: 0,
        sublayer: 'ffn_add_node'
    };
}
