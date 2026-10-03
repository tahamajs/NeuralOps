/**
 * 3d/scene_router.js
 * ==================
 * Multi-Category 3D Scene Router & Custom Procedural Geometry Builder.
 * Generates mathematically-accurate 3D scene graphs, meshes, and animation timelines
 * tailored specifically for each of the 9 Architectural Categories (A through I).
 */

import { COLORS } from './visual_language.js';

/**
 * Builds scene specification object for a given module and category.
 * Can run in pure ESM (Node.js tests) or Browser (Three.js WebGL).
 */
export function buildSceneSpec(moduleId, category = 'A', moduleData = {}) {
    const spec = {
        moduleId,
        category,
        nodes: [],
        connections: [],
        features: {
            hasSquareMatrix: false,
            hasStateRibbon: false,
            hasSingularSpectrum: false,
            hasRouterGating: false,
            hasQuantizationBins: false,
            hasBandedWindow: false,
            hasAssociativeReorder: false,
            hasSparseMask: false
        }
    };

    switch (category) {
        case 'D': // Recurrent / Linear (HGRN2, Mamba, RWKV, DeltaNet)
            // STRICTLY NO S×S square matrix, NO Q/K/V
            spec.features.hasStateRibbon = true;
            spec.nodes = [
                { id: 'input_stream', type: 'sequence', name: 'Input Sequence X', color: COLORS.input },
                { id: 'forget_gate', type: 'gate', name: 'Forget Gate f_t', color: COLORS.gate },
                { id: 'input_gate', type: 'gate', name: 'Input Gate i_t', color: COLORS.gate },
                { id: 'value_proj', type: 'projection', name: 'Value Projection v_t', color: COLORS.value },
                { id: 'state_ribbon', type: 'recurrent_state', name: 'Latent Recurrent Ribbon h_t', color: COLORS.state },
                { id: 'output_y', type: 'output', name: 'Contextualized Output Y', color: COLORS.output }
            ];
            spec.connections = [
                { from: 'input_stream', to: 'forget_gate' },
                { from: 'input_stream', to: 'input_gate' },
                { from: 'input_stream', to: 'value_proj' },
                { from: 'forget_gate', to: 'state_ribbon' },
                { from: 'input_gate', to: 'state_ribbon' },
                { from: 'value_proj', to: 'state_ribbon' },
                { from: 'state_ribbon', to: 'output_y' }
            ];
            break;

        case 'E': // SVD / Low-Rank (A3, ASVD, SVD-LLM, LoRA)
            spec.features.hasSingularSpectrum = true;
            spec.nodes = [
                { id: 'dense_w', type: 'weight_block', name: 'Original Dense Matrix W [d × d]', color: COLORS.attention },
                { id: 'singular_values', type: 'spectrum_bars', name: 'Singular Value Spectrum Σ', color: COLORS.expert },
                { id: 'u_factor', type: 'svd_u', name: 'Left Factor U_r [d × r]', color: COLORS.reconstructed },
                { id: 'sigma_factor', type: 'svd_sigma', name: 'Truncated Singular Diagonal Σ_r [r × r]', color: COLORS.expert },
                { id: 'v_factor', type: 'svd_v', name: 'Right Factor V_r^T [r × d]', color: COLORS.query },
                { id: 'reconstructed_wr', type: 'reconstructed_w', name: 'Low-Rank Reconstruction W_r', color: COLORS.reconstructed },
                { id: 'residual_error', type: 'error_heatmap', name: 'Residual Error |W - W_r|', color: COLORS.error_positive }
            ];
            spec.connections = [
                { from: 'dense_w', to: 'singular_values' },
                { from: 'singular_values', to: 'u_factor' },
                { from: 'singular_values', to: 'sigma_factor' },
                { from: 'singular_values', to: 'v_factor' },
                { from: 'u_factor', to: 'reconstructed_wr' },
                { from: 'v_factor', to: 'reconstructed_wr' },
                { from: 'reconstructed_wr', to: 'residual_error' }
            ];
            break;

        case 'B': // Sparse / Approximate (Performer, Linformer, Nystrom)
            // NO S×S square matrix
            spec.features.hasAssociativeReorder = true;
            spec.nodes = [
                { id: 'input_x', type: 'sequence', name: 'Input Sequence X', color: COLORS.input },
                { id: 'phi_q', type: 'kernel_features', name: 'Random Feature Map φ(Q) [S × m]', color: COLORS.query },
                { id: 'phi_k', type: 'kernel_features', name: 'Random Feature Map φ(K) [S × m]', color: COLORS.key },
                { id: 'v_slab', type: 'projection', name: 'Value Tensor V [S × d]', color: COLORS.value },
                { id: 'kv_buffer', type: 'small_buffer', name: 'Low-Dim Buffer M = φ(K)^T V [m × d]', color: COLORS.reconstructed },
                { id: 'out_slab', type: 'output', name: 'Associative Output = φ(Q) M [S × d]', color: COLORS.output }
            ];
            spec.connections = [
                { from: 'phi_k', to: 'kv_buffer' },
                { from: 'v_slab', to: 'kv_buffer' },
                { from: 'phi_q', to: 'out_slab' },
                { from: 'kv_buffer', to: 'out_slab' }
            ];
            break;

        case 'C': // Sliding Window (Longformer, Local Window)
            spec.features.hasSquareMatrix = true;
            spec.features.hasBandedWindow = true;
            spec.nodes = [
                { id: 'q_slab', type: 'projection', name: 'Query Q', color: COLORS.query },
                { id: 'k_slab', type: 'projection', name: 'Key K', color: COLORS.key },
                { id: 'banded_matrix', type: 'banded_square', name: 'Banded S×S Window (Width W)', color: COLORS.attention },
                { id: 'v_slab', type: 'projection', name: 'Value V', color: COLORS.value },
                { id: 'out_slab', type: 'output', name: 'Local Context Output', color: COLORS.output }
            ];
            spec.connections = [
                { from: 'q_slab', to: 'banded_matrix' },
                { from: 'k_slab', to: 'banded_matrix' },
                { from: 'banded_matrix', to: 'out_slab' },
                { from: 'v_slab', to: 'out_slab' }
            ];
            break;

        case 'F': // MoE (Mixture of Experts)
            spec.features.hasRouterGating = true;
            spec.nodes = [
                { id: 'token_input', type: 'sequence', name: 'Input Tokens X', color: COLORS.input },
                { id: 'moe_router', type: 'router', name: 'Top-K Softmax Router', color: COLORS.router },
                { id: 'expert_1', type: 'expert', name: 'Expert Network 1 (FFN)', color: COLORS.expert },
                { id: 'expert_2', type: 'expert', name: 'Expert Network 2 (FFN)', color: COLORS.expert },
                { id: 'expert_3', type: 'expert', name: 'Expert Network 3 (FFN)', color: COLORS.expert },
                { id: 'expert_4', type: 'expert', name: 'Expert Network 4 (FFN)', color: COLORS.expert },
                { id: 'gated_output', type: 'output', name: 'Weighted Sum Output', color: COLORS.output }
            ];
            spec.connections = [
                { from: 'token_input', to: 'moe_router' },
                { from: 'moe_router', to: 'expert_1' },
                { from: 'moe_router', to: 'expert_2' },
                { from: 'expert_1', to: 'gated_output' },
                { from: 'expert_2', to: 'gated_output' }
            ];
            break;

        case 'G': // Pruning
            spec.features.hasSparseMask = true;
            spec.nodes = [
                { id: 'dense_w', type: 'weight_block', name: 'Dense Weights W', color: COLORS.attention },
                { id: 'pruning_mask', type: 'sparse_mask', name: 'Sparsity Mask M (Top 50%)', color: COLORS.zero },
                { id: 'sparse_w', type: 'sparse_weight', name: 'Pruned Sparse Weights W_sparse', color: COLORS.key }
            ];
            spec.connections = [
                { from: 'dense_w', to: 'pruning_mask' },
                { from: 'pruning_mask', to: 'sparse_w' }
            ];
            break;

        case 'H': // Quantization
            spec.features.hasQuantizationBins = true;
            spec.nodes = [
                { id: 'fp32_weights', type: 'histogram', name: 'FP32 Continuous Weights', color: COLORS.input },
                { id: 'quant_grid', type: 'quant_bins', name: 'Uniform 8-Bit Grid [-128..127]', color: COLORS.gate },
                { id: 'int8_weights', type: 'discretized', name: 'INT8 Discretized Weights', color: COLORS.reconstructed }
            ];
            spec.connections = [
                { from: 'fp32_weights', to: 'quant_grid' },
                { from: 'quant_grid', to: 'int8_weights' }
            ];
            break;

        case 'I': // Hybrid
            spec.features.hasStateRibbon = true;
            spec.features.hasSquareMatrix = true;
            spec.nodes = [
                { id: 'input_x', type: 'sequence', name: 'Input X', color: COLORS.input },
                { id: 'attn_branch', type: 'attention_block', name: 'Transformer Attention Branch', color: COLORS.attention },
                { id: 'mamba_branch', type: 'recurrent_block', name: 'Selective State Space Branch', color: COLORS.state },
                { id: 'fusion_gate', type: 'fusion', name: 'Adaptive Gated Fusion Node', color: COLORS.router },
                { id: 'hybrid_out', type: 'output', name: 'Fused Representation', color: COLORS.output }
            ];
            spec.connections = [
                { from: 'input_x', to: 'attn_branch' },
                { from: 'input_x', to: 'mamba_branch' },
                { from: 'attn_branch', to: 'fusion_gate' },
                { from: 'mamba_branch', to: 'fusion_gate' },
                { from: 'fusion_gate', to: 'hybrid_out' }
            ];
            break;

        case 'A': // Dense Attention (Default)
        default:
            spec.features.hasSquareMatrix = true;
            spec.nodes = [
                { id: 'q_proj', type: 'projection', name: 'Query Projection W_Q', color: COLORS.query },
                { id: 'k_proj', type: 'projection', name: 'Key Projection W_K', color: COLORS.key },
                { id: 'v_proj', type: 'projection', name: 'Value Projection W_V', color: COLORS.value },
                { id: 'attn_square', type: 'dense_square', name: 'Pairwise S × S Attention Square', color: COLORS.attention },
                { id: 'out_proj', type: 'projection', name: 'Output Projection W_O', color: COLORS.output }
            ];
            spec.connections = [
                { from: 'q_proj', to: 'attn_square' },
                { from: 'k_proj', to: 'attn_square' },
                { from: 'attn_square', to: 'out_proj' },
                { from: 'v_proj', to: 'out_proj' }
            ];
            break;
    }

    return spec;
}

/**
 * Derives the animation timeline for a specific module category.
 */
export function getTimelineForCategory(category = 'A') {
    switch (category) {
        case 'D': // Recurrent
            return [
                { step: 1, name: 'Token Ingress & Linear Projections', duration: 0.8 },
                { step: 2, name: 'Compute Forget & Input Gates', duration: 1.0 },
                { step: 3, name: 'State Space Recurrence Ribbon Update', duration: 1.5 },
                { step: 4, name: 'Output Normalization & Residual Add', duration: 0.7 }
            ];

        case 'E': // SVD
            return [
                { step: 1, name: 'Dense Weight Matrix Spectrum Ingress', duration: 0.8 },
                { step: 2, name: 'Singular Value Decomposition (U Σ V^T)', duration: 1.2 },
                { step: 3, name: 'Rank-r Truncation & Dimension Pinch', duration: 1.4 },
                { step: 4, name: 'Low-Rank Matrix Reconstruction', duration: 1.0 },
                { step: 5, name: 'Pointwise Residual Error Analysis', duration: 0.8 }
            ];

        case 'F': // MoE
            return [
                { step: 1, name: 'Token Router Logits Calculation', duration: 0.8 },
                { step: 2, name: 'Top-K Softmax Dispatching', duration: 1.2 },
                { step: 3, name: 'Parallel Expert Network Forward Pass', duration: 1.4 },
                { step: 4, name: 'Weighted Expert Fusion & Combine', duration: 0.9 }
            ];

        case 'B': // Sparse / Approximate
            return [
                { step: 1, name: 'Random Kernel Feature Projection φ(Q), φ(K)', duration: 1.0 },
                { step: 2, name: 'Associative Buffer Materialization M = φ(K)^T V', duration: 1.4 },
                { step: 3, name: 'Associative Multiply Output = φ(Q) M', duration: 1.2 }
            ];

        default: // Dense / C / G / H / I
            return [
                { step: 1, name: 'Linear Projections (Q, K, V)', duration: 0.8 },
                { step: 2, name: 'Attention Score Computation (Q · K^T)', duration: 1.2 },
                { step: 3, name: 'Variance Scaling & Softmax Normalization', duration: 1.0 },
                { step: 4, name: 'Context Aggregation (A · V) & Projection', duration: 1.0 }
            ];
    }
}
