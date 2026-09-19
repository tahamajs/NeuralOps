/**
 * react/types/module.types.js
 * ===========================
 * Canonical JavaScript state definitions, shape signatures, and substate mappings.
 */

export const SVD_STATES = {
  idle: {
    title: "Original Dense Weight Matrix",
    formula: "W \\in \\mathbb{R}^{d_{out} \\times d_{in}}",
    description: "The full uncompressed weight tensor representing the linear attention projection.",
    shapes: ["768×768"],
    highlightNodes: ["weight_block"]
  },
  show_W: {
    title: "Weight Matrix Saliency",
    formula: "W = [w_{ij}]",
    description: "Weight magnitude matrix prior to singular value factorization.",
    shapes: ["768×768"],
    highlightNodes: ["weight_block"]
  },
  decompose: {
    title: "Full SVD Decomposition",
    formula: "W = U \\Sigma V^\\top",
    description: "Exact singular value decomposition separating orthogonal spatial orientations U, V from singular values Σ.",
    shapes: ["768×768", "768×768", "768×768"],
    highlightNodes: ["U_block", "Sigma_block", "V_block"]
  },
  singular_values: {
    title: "Singular Value Spectrum",
    formula: "\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_{768} > 0",
    description: "Log-scale eigenvalue spectrum. Rapid singular decay demonstrates intrinsic low-rank subspace.",
    shapes: ["768"],
    highlightNodes: ["sv_bars"]
  },
  truncate: {
    title: "Rank-r Truncation",
    formula: "W_r = U_r \\Sigma_r V_r^\\top",
    description: "Eckart-Young-Mirsky optimal rank-r approximation keeping only the top r singular values.",
    shapes: ["768×r", "r×r", "r×768"],
    highlightNodes: ["U_r", "Sigma_r", "V_r"]
  },
  reconstruct: {
    title: "Factored Reconstruction",
    formula: "W_r = (U_r \\sqrt{\\Sigma_r}) (\\sqrt{\\Sigma_r} V_r^\\top)",
    description: "Symmetric absorption of singular scales into dual low-rank bottleneck factors.",
    shapes: ["768×768"],
    highlightNodes: ["Wr_block"]
  },
  error_heatmap: {
    title: "Reconstruction Error",
    formula: "E = |W - W_r|, \\quad \\|E\\|_F = \\sqrt{\\sum_{i=r+1}^{d} \\sigma_i^2}",
    description: "Residual approximation error showing minimal loss in dominant semantic directions.",
    shapes: ["768×768"],
    highlightNodes: ["error_slab"]
  }
};

export const RECURRENT_STATES = {
  idle: {
    title: "Initial Recurrent State",
    formula: "h_0 = \\mathbf{0} \\in \\mathbb{R}^d",
    description: "System begins at rest with zero state buffer before encountering the token sequence.",
    shapes: ["768"],
    highlightNodes: ["state_buffer_0"],
    forgetGate: 1.0,
    inputGate: 0.0
  },
  t0: {
    title: "Initial State (t=0)",
    formula: "h_0 = \\mathbf{0}",
    description: "Hidden state memory cleared to zero vector.",
    shapes: ["768"],
    highlightNodes: ["h0"],
    forgetGate: 1.0,
    inputGate: 0.0
  },
  t1: {
    title: "Token 1 Transition (t=1)",
    formula: "h_1 = f_1 \\odot h_0 + (1 - f_1) \\odot (i_1 \\odot v_1)",
    description: "First token injects information into the recurrent memory vector via input gate i1.",
    shapes: ["768"],
    highlightNodes: ["h1"],
    forgetGate: 0.92,
    inputGate: 0.35
  },
  t2: {
    title: "Token 2 Transition (t=2)",
    formula: "h_2 = f_2 \\odot h_1 + (1 - f_2) \\odot (i_2 \\odot v_2)",
    description: "Memory update with elementwise forgetting of earlier state and selective incorporation of token 2.",
    shapes: ["768"],
    highlightNodes: ["h2"],
    forgetGate: 0.88,
    inputGate: 0.42
  },
  t3: {
    title: "Token 3 Transition (t=3)",
    formula: "h_3 = f_3 \\odot h_2 + (1 - f_3) \\odot (i_3 \\odot v_3)",
    description: "Sequential accumulation maintaining strict O(1) space complexity without pairwise attention.",
    shapes: ["768"],
    highlightNodes: ["h3"],
    forgetGate: 0.85,
    inputGate: 0.48
  },
  t4: {
    title: "Token 4 Transition (t=4)",
    formula: "h_4 = f_4 \\odot h_3 + (1 - f_4) \\odot (i_4 \\odot v_4)",
    description: "Fourth token updates persistent memory trunk ready for immediate output readout.",
    shapes: ["768"],
    highlightNodes: ["h4"],
    forgetGate: 0.81,
    inputGate: 0.55
  },
  output: {
    title: "Linear Readout Projection",
    formula: "y_t = W_o (g_t \\odot h_t)",
    description: "Output projection computed directly from the current recurrent state vector.",
    shapes: ["768"],
    highlightNodes: ["output_proj"],
    forgetGate: 0.80,
    inputGate: 0.50
  }
};

export const DENSE_STATES = {
  idle: {
    title: "Token Input Embeddings",
    formula: "X \\in \\mathbb{R}^{S \\times d}",
    description: "Sequence tokens entering multi-head attention.",
    shapes: ["S×768"],
    highlightNodes: ["input_tokens"]
  },
  qkv_projection: {
    title: "Q, K, V Projections",
    formula: "Q = XW_Q, \\; K = XW_K, \\; V = XW_V",
    description: "Linear transformations mapping hidden states into distinct query, key, and value head subspaces.",
    shapes: ["S×64", "S×64", "S×64"],
    highlightNodes: ["q_proj", "k_proj", "v_proj"]
  },
  score_matrix: {
    title: "Pairwise Score Matrix",
    formula: "A = \\frac{QK^\\top}{\\sqrt{d_k}} + M_{\\text{causal}}",
    description: "Full S×S dot-product attention scores measuring all-to-all token relationships.",
    shapes: ["S×S"],
    highlightNodes: ["score_square"]
  },
  softmax: {
    title: "Softmax Probability Matrix",
    formula: "P = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\right)",
    description: "Row-normalized attention probabilities where each token's weights sum to exactly 1.",
    shapes: ["S×S"],
    highlightNodes: ["prob_square"]
  },
  context: {
    title: "Value Aggregation",
    formula: "O = PV",
    description: "Weighted aggregation of value vectors across context tokens.",
    shapes: ["S×64"],
    highlightNodes: ["context_matrix"]
  },
  output: {
    title: "Multi-Head Output Projection",
    formula: "Y = O W_O",
    description: "Final linear projection synthesizing heads back to hidden dimension.",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const SPARSE_STATES = {
  idle: {
    title: "Input Tokens",
    formula: "X \\in \\mathbb{R}^{S \\times d}",
    description: "Sequence tokens for linear kernelized attention.",
    shapes: ["S×768"],
    highlightNodes: ["input_tokens"]
  },
  feature_map: {
    title: "Non-linear Feature Mapping",
    formula: "\\phi(Q), \\; \\phi(K) \\in \\mathbb{R}^{S \\times m}",
    description: "Random Fourier or positive orthogonal kernel features bypassing the quadratic matrix.",
    shapes: ["S×128", "S×128"],
    highlightNodes: ["feature_map"]
  },
  associative_reorder: {
    title: "Associative Multiplication",
    formula: "O = \\phi(Q) \\left(\\phi(K)^\\top V\\right)",
    description: "Parenthesization reordering reduces complexity from O(S²d) to O(S m d).",
    shapes: ["128×64", "S×64"],
    highlightNodes: ["reorder_block"]
  },
  output: {
    title: "Linear Attention Output",
    formula: "Y = O W_O",
    description: "Projected output representation with linear sequence complexity.",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const WINDOW_STATES = {
  idle: {
    title: "Sequence Tokens",
    formula: "X \\in \\mathbb{R}^{S \\times d}",
    description: "Tokens entering banded neighborhood attention.",
    shapes: ["S×768"],
    highlightNodes: ["input_tokens"]
  },
  qkv_projection: {
    title: "Q, K, V Projections",
    formula: "Q = XW_Q, \\; K = XW_K, \\; V = XW_V",
    description: "Linear projections mapped into head subspaces.",
    shapes: ["S×64", "S×64", "S×64"],
    highlightNodes: ["qkv"]
  },
  banded_mask: {
    title: "Banded Context Mask",
    formula: "M_{ij} = \\begin{cases} 0 & |i - j| \\leq w \\\\ -\\infty & \\text{otherwise} \\end{cases}",
    description: "Restricting attention reception field to local window width w.",
    shapes: ["S×S"],
    highlightNodes: ["banded_mask"]
  },
  local_attention: {
    title: "Local Window Attention",
    formula: "O_i = \\sum_{j = i - w}^{i + w} P_{ij} V_j",
    description: "O(S × w) bounded complexity compute within receptive field.",
    shapes: ["S×64"],
    highlightNodes: ["local_attn"]
  },
  output: {
    title: "Projected Local Output",
    formula: "Y = O W_O",
    description: "Final projection back to model hidden dimension.",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const MOE_STATES = {
  idle: {
    title: "Input Representations",
    formula: "X \\in \\mathbb{R}^{S \\times d}",
    description: "Tokens ready for sparse expert routing.",
    shapes: ["S×768"],
    highlightNodes: ["tokens"]
  },
  router_gate: {
    title: "Router Gate Distribution",
    formula: "G(x) = \\text{TopK}(\\text{Softmax}(x W_g + \\epsilon), k)",
    description: "Learned gating network evaluating affinity scores for each candidate expert.",
    shapes: ["S×8"],
    highlightNodes: ["router_gate"]
  },
  expert_dispatch: {
    title: "Token Dispatch",
    formula: "\\mathcal{B}_e = \\{ x_i \\mid e \\in \\text{argtopk}(G(x_i)) \\}",
    description: "Routing selected tokens into dedicated expert batches subject to capacity constraints.",
    shapes: ["C×768"],
    highlightNodes: ["dispatch_mesh"]
  },
  expert_compute: {
    title: "Expert FFN Computation",
    formula: "E_e(x) = W_{e,2} \\cdot \\text{GELU}(W_{e,1} x)",
    description: "Parallel sub-network evaluation across activated experts.",
    shapes: ["C×3072"],
    highlightNodes: ["expert_blocks"]
  },
  expert_combine: {
    title: "Expert Output Combination",
    formula: "y = \\sum_{e \\in \\text{TopK}} G(x)_e E_e(x)",
    description: "Weighted aggregation of expert outputs based on gate scores.",
    shapes: ["S×768"],
    highlightNodes: ["combine_node"]
  },
  output: {
    title: "Routed MoE Output",
    formula: "Y = y + X",
    description: "Residual combination of routed representations.",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const PRUNING_STATES = {
  idle: {
    title: "Unpruned Dense Weight",
    formula: "W \\in \\mathbb{R}^{d_{out} \\times d_{in}}",
    description: "100% dense weight matrix before importance evaluation.",
    shapes: ["768×768"],
    highlightNodes: ["weight_block"]
  },
  importance_eval: {
    title: "Saliency Importance Scoring",
    formula: "S_{ij} = |W_{ij}| \\cdot \\|X_{:j}\\|_2",
    description: "Second-order Taylor or Wanda importance estimating parameter contribution to loss.",
    shapes: ["768×768"],
    highlightNodes: ["saliency_map"]
  },
  mask_generation: {
    title: "Binary Sparsity Mask",
    formula: "M_{ij} = \\mathbb{I}(S_{ij} \\geq \\tau)",
    description: "Thresholding top-p parameters to retain while zeroing unimportant weights.",
    shapes: ["768×768"],
    highlightNodes: ["sparsity_mask"]
  },
  pruned_projection: {
    title: "Sparse Forward Multiply",
    formula: "Y = (W \\odot M) X",
    description: "Accelerated sparse matrix-vector multiplication with zeroed channels.",
    shapes: ["S×768"],
    highlightNodes: ["pruned_gemm"]
  },
  output: {
    title: "Pruned Output Representation",
    formula: "Y \\in \\mathbb{R}^{S \\times d_{out}}",
    description: "Model features preserved at target sparsity level (e.g. 50% 2:4 structured).",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const QUANT_STATES = {
  idle: {
    title: "Full Precision Weights (FP16)",
    formula: "W \\in \\mathbb{R}^{d_{out} \\times d_{in}}",
    description: "Unquantized floating-point weight tensors.",
    shapes: ["768×768"],
    highlightNodes: ["fp16_block"]
  },
  calibration: {
    title: "Activation Calibration Profiling",
    formula: "[\\alpha, \\beta] = [\\min(X), \\max(X)]",
    description: "Activation dynamic range calibration to prevent outlier clipping.",
    shapes: ["2"],
    highlightNodes: ["calib_meter"]
  },
  scale_calc: {
    title: "Scale & Zero-Point Derivation",
    formula: "s = \\frac{\\max(W) - \\min(W)}{2^b - 1}, \\quad z = -\\text{round}(\\min(W) / s)",
    description: "Dynamic per-channel or block-wise scaling factor calculation.",
    shapes: ["768"],
    highlightNodes: ["scale_vector"]
  },
  quantize_int8_fp8: {
    title: "Quantized Bit Packing (FP8/INT8)",
    formula: "W_q = \\text{clamp}\\left(\\text{round}\\left(\\frac{W}{s}\\right) + z, -128, 127\\right)",
    description: "Compressed integer or E4M3 FP8 representation reducing memory bandwidth by 2x-4x.",
    shapes: ["768×768 (8-bit)"],
    highlightNodes: ["packed_matrix"]
  },
  dequantize: {
    title: "Dequantize & Accumulate",
    formula: "\\hat{Y} = s_x s_w (X_q W_q^\\top)",
    description: "High-precision tensor core integer GEMM with scalar scaling.",
    shapes: ["S×768"],
    highlightNodes: ["gemm_core"]
  },
  output: {
    title: "Quantized Layer Output",
    formula: "Y = \\hat{Y}",
    description: "High-throughput output with imperceptible perplexity delta.",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const HYBRID_STATES = {
  idle: {
    title: "Input Hidden States",
    formula: "X \\in \\mathbb{R}^{S \\times d}",
    description: "Input sequence entering hybrid layered architecture.",
    shapes: ["S×768"],
    highlightNodes: ["input_tokens"]
  },
  layer_route: {
    title: "Hybrid Interleaving Dispatch",
    formula: "\\text{Schedule}(l) = \\begin{cases} \\text{Mamba / SSM} & l \\equiv 0 \\pmod 2 \\\\ \\text{Dense Attn} & l \\equiv 1 \\pmod 2 \\end{cases}",
    description: "Alternating sequence schedule balancing linear inference with global recall.",
    shapes: ["L"],
    highlightNodes: ["schedule_ring"]
  },
  hybrid_ssm_step: {
    title: "Recurrent SSM Phase",
    formula: "h_t = \\bar{A} h_{t-1} + \\bar{B} x_t, \\quad y_{\\text{ssm}} = C h_t",
    description: "Subquadratic linear token updates capturing local syntax and unbounded context.",
    shapes: ["768"],
    highlightNodes: ["ssm_trunk"]
  },
  hybrid_attn_step: {
    title: "Global Attention Phase",
    formula: "y_{\\text{attn}} = \\text{Softmax}\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right) V",
    description: "Periodic global self-attention layer resolving needle-in-a-haystack retrieval.",
    shapes: ["S×768"],
    highlightNodes: ["attn_square"]
  },
  fusion_gate: {
    title: "Gated Representation Fusion",
    formula: "y = \\sigma(W_g x) \\odot y_{\\text{ssm}} + (1 - \\sigma(W_g x)) \\odot y_{\\text{attn}}",
    description: "Learned gating parameter blending SSM and attention representations.",
    shapes: ["S×768"],
    highlightNodes: ["fusion_gate"]
  },
  output: {
    title: "Composite Hybrid Output",
    formula: "Y = y + X",
    description: "Residual hybrid output providing best-of-both-worlds compute profile.",
    shapes: ["S×768"],
    highlightNodes: ["output_block"]
  }
};

export const SUBSTATE_MAP = {
  "E/fisher_svd": ["fisher_weights_visible", "ln_recalibration_visible"],
  "E/plain_svd": ["no_weighting"],
  "E/adaptive_rank": ["energy_threshold_visible"],
  "A/grouped_query": ["kv_sharing_visible", "kv_sharing_collapsed"],
  "A/flash_attention": ["tiling_blocks_visible", "online_softmax_visible"],
  "A/multi_head": ["head_split_visible"],
  "B/performer": ["kernel_features_visible"],
  "B/linformer": ["low_rank_projection_visible"],
  "B/nystrom": ["landmark_selection_visible"],
  "B/lsh": ["hash_buckets_visible"],
  "C/sliding_window": ["local_window_mask_visible"],
  "C/adaptive_span": ["dynamic_span_visible"],
  "C/dilated_window": ["dilation_stride_visible"],
  "D/recurrent_elementwise": ["forget_gate_visible", "input_gate_visible"],
  "D/state_space": ["state_space_visible", "selective_scan_visible"],
  "D/delta_net": ["delta_rule_visible"],
  "F/expert_choice": ["capacity_limit_visible"],
  "F/topk_routing": ["routing_probabilities_visible"],
  "G/magnitude": ["threshold_cutoff_visible"],
  "G/wanda": ["activation_weight_product_visible"],
  "G/sparsegpt": ["hessian_update_visible"],
  "H/fp8_e4m3": ["dynamic_scale_visible"],
  "H/int8_smooth": ["smoothing_factors_visible"],
  "I/interleaved": ["layer_index_indicator_visible"],
  "I/fused": ["cross_layer_fusion_visible"]
};
