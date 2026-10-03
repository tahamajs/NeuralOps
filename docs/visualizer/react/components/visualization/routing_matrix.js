/**
 * react/components/visualization/routing_matrix.js
 * ===============================================
 * Verified routing table and forbidden state invariants for all 252 catalog modules.
 */

export const ROUTING_MATRIX = {
  "SOAFusedAttention": {
    "id": "SOAFusedAttention",
    "name": "SOA Fused Attention: Low-Rank Latent Space Multi-Head Attention w",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "adaptive_span_attention": {
    "id": "adaptive_span_attention",
    "name": "Adaptive Span Attention: Dynamic Head-Specific Context Window Gat",
    "category": "C",
    "subcategory": "sliding_window",
    "states": [
      "idle",
      "qkv_projection",
      "banded_mask",
      "local_attention",
      "output"
    ],
    "substates": [
      "dynamic_span_visible"
    ],
    "defaultState": "banded_mask",
    "forbidden": [
      "global_cross_attention",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "adaptive_span_inject": {
    "id": "adaptive_span_inject",
    "name": "Adaptive Span Attention: Dynamic Head-Specific Context Window Gat",
    "category": "C",
    "subcategory": "sliding_window",
    "states": [
      "idle",
      "qkv_projection",
      "banded_mask",
      "local_attention",
      "output"
    ],
    "substates": [
      "dynamic_span_visible"
    ],
    "defaultState": "banded_mask",
    "forbidden": [
      "global_cross_attention",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "adaptive_sparse_attention": {
    "id": "adaptive_sparse_attention",
    "name": "Adaptive Sparse Attention: Dynamic Top-K & Threshold-Based Sparsi",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "adaptive_sparse_inject": {
    "id": "adaptive_sparse_inject",
    "name": "Adaptive Sparse Attention: Dynamic Top-K & Threshold-Based Sparsi",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "adaptive_svd_attention": {
    "id": "adaptive_svd_attention",
    "name": "Adaptive SVD Attention: Low-Rank SVD Projections with Parallel Re",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "adasplash_attention": {
    "id": "adasplash_attention",
    "name": "AdaSplash Attention: Adaptive Splash Attention with alpha-Entmax ",
    "category": "C",
    "subcategory": "sliding_window",
    "states": [
      "idle",
      "qkv_projection",
      "banded_mask",
      "local_attention",
      "output"
    ],
    "substates": [
      "local_window_mask_visible"
    ],
    "defaultState": "banded_mask",
    "forbidden": [
      "global_cross_attention",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "adasplash_inject": {
    "id": "adasplash_inject",
    "name": "AdaSplash Attention: Adaptive Splash Attention with alpha-Entmax ",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "aegis_hybrid": {
    "id": "aegis_hybrid",
    "name": "AEGIS-Hybrid Attention: Low-Rank Factorized Hybrid Attention with",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aegis_hybrid_attention": {
    "id": "aegis_hybrid_attention",
    "name": "Compatibility shim for aegis_hybrid_attention.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "aegis_spectron_fp8": {
    "id": "aegis_spectron_fp8",
    "name": "AEGIS-Spectron-FP8: Tri-Branch Hybrid Attention with Token-Level ",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "dynamic_scale_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "aegis_v1": {
    "id": "aegis_v1",
    "name": "AEGIS-Attention V1: Adaptive Enhanced Global-Local Integrated Spa",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aegis_v2": {
    "id": "aegis_v2",
    "name": "AEGIS-Attention V2: Low-Rank Factorized Hybrid Attention with Ada",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aegis_v3": {
    "id": "aegis_v3",
    "name": "AEGIS-Attention V3: Adaptive Enhanced Global-Local Integrated Spa",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aegis_v4": {
    "id": "aegis_v4",
    "name": "AEGIS-Attention V4: Adaptive Enhanced Global-Local Integrated Spa",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aegis_v5": {
    "id": "aegis_v5",
    "name": "AEGIS-Attention V5: Adaptive Enhanced Global-Local Integrated Spa",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aegis_v6": {
    "id": "aegis_v6",
    "name": "AEGIS-Attention V6: Unified SVD-LoRA Hybrid Attention with Orthog",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "asvd_attention": {
    "id": "asvd_attention",
    "name": "ASVD Attention: Activation-Aware Singular Value Decomposition Arc",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "bam_umoe": {
    "id": "bam_umoe",
    "name": "BAM-uMoE: Bidirectional Attentive Mixture-of-Unified-Experts",
    "category": "F",
    "subcategory": "mixture_of_experts",
    "states": [
      "idle",
      "router_gate",
      "expert_dispatch",
      "expert_compute",
      "expert_combine",
      "output"
    ],
    "substates": [
      "routing_probabilities_visible"
    ],
    "defaultState": "router_gate",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "block_transformer_hiba": {
    "id": "block_transformer_hiba",
    "name": "Block Transformer with Hierarchical Intra-Block and Inter-Block A",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "chela_slay": {
    "id": "chela_slay",
    "name": "CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "cluster_attention": {
    "id": "cluster_attention",
    "name": "ClusterFormer: Dynamic Token-Clustering Attention via Batch Itera",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "cluster_inject": {
    "id": "cluster_inject",
    "name": "ClusterFormer: Dynamic Token-Clustering Attention via Batch Itera",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "compressed_attention": {
    "id": "compressed_attention",
    "name": "Compressed Multi-Head Attention via Truncated Singular Value Deco",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "compressed_mha": {
    "id": "compressed_mha",
    "name": "Compressed Multi-Head Attention (Compressed-MHA)",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "compressed_mha_attention": {
    "id": "compressed_mha_attention",
    "name": "Compressed Multi-Head Attention for DistilBERT and Multi-Architec",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "cosformer": {
    "id": "cosformer",
    "name": "Cosformer: Rethinking Softmax in Attention with Linear Complexity",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "cosformer_inject": {
    "id": "cosformer_inject",
    "name": "Deprecated stub. Re-exports base module for registry compatibilit",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "decay_linear_attention": {
    "id": "decay_linear_attention",
    "name": "Decay Linear Attention (DecayLinearAttention).",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "eat": {
    "id": "eat",
    "name": "EAT: Efficient Attention Transformer with Progressive Token Pruni",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "expert_choice_moe": {
    "id": "expert_choice_moe",
    "name": "Expert Choice Mixture of Experts (EC-MoE) Attention Block",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "factorised_matrix_state": {
    "id": "factorised_matrix_state",
    "name": "Factorised Matrix State Attention: Bidirectional Associative Matr",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "flash_attention3": {
    "id": "flash_attention3",
    "name": "FlashAttention-3: Fast and Memory-Efficient Exact Attention with ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "flash_attention3_fp8": {
    "id": "flash_attention3_fp8",
    "name": "FlashAttention-3 FP8: Low-Precision 8-Bit Float Transformer Atten",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "full_attention": {
    "id": "full_attention",
    "name": "Full Attention: Standard Multi-Head Scaled Dot-Product Attention ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "fused_svd_kernel": {
    "id": "fused_svd_kernel",
    "name": "Fused SVD-Linear Layer with SRAM-Resident RMSNorm & Dual-Projecti",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "fusion_attn": {
    "id": "fusion_attn",
    "name": "FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "galerkin": {
    "id": "galerkin",
    "name": "Petrov-Galerkin Projection Attention: Energy-Preserving Spectral ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "galerkin_attention": {
    "id": "galerkin_attention",
    "name": "Petrov-Galerkin Projection Attention: Energy-Preserving Spectral ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "galerkin_inject": {
    "id": "galerkin_inject",
    "name": "Petrov-Galerkin Projection Attention: Energy-Preserving Spectral ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "gated_delta_net": {
    "id": "gated_delta_net",
    "name": "Gated DeltaNet: Multi-Head Bidirectional Delta Rule Linear Recurr",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "delta_rule_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "gated_deltanet_block": {
    "id": "gated_deltanet_block",
    "name": "Gated DeltaNet Block: Hybrid Multi-Head Attention and Recurrent D",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "delta_rule_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "gated_linear_attention": {
    "id": "gated_linear_attention",
    "name": "Gated Linear Attention (GatedLinearAttention).",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "goldfinch_ahn": {
    "id": "goldfinch_ahn",
    "name": "GoldFinch-AHN: Adaptive Hybrid Network Combining Sliding Window a",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "gqa": {
    "id": "gqa",
    "name": "Grouped-Query Attention (GQA): Memory-Efficient Shared Key-Value ",
    "category": "A",
    "subcategory": "grouped_query",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [
      "kv_sharing_visible"
    ],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "gqa_attention": {
    "id": "gqa_attention",
    "name": "Grouped-Query Attention (GQA): Memory-Efficient Shared Key-Value ",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [
      "kv_sharing_visible"
    ],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "granite_hybrid": {
    "id": "granite_hybrid",
    "name": "Granite-Hybrid Attention: Interleaved Mamba-2 State Space Archite",
    "category": "I",
    "subcategory": "hybrid_composite",
    "states": [
      "idle",
      "layer_route",
      "hybrid_ssm_step",
      "hybrid_attn_step",
      "fusion_gate",
      "output"
    ],
    "substates": [
      "layer_index_indicator_visible"
    ],
    "defaultState": "hybrid_ssm_step",
    "forbidden": [
      "pure_dense_only",
      "svd_singular_values"
    ]
  },
  "griffin_ssd": {
    "id": "griffin_ssd",
    "name": "Griffin SSD: Hybrid Local Sliding-Window Attention and Global Sta",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "halo_regla": {
    "id": "halo_regla",
    "name": "HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "hamming_attention": {
    "id": "hamming_attention",
    "name": "Hamming Attention Distillation (HAD): Sign-Binarized Attention wi",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "hgrn2_gated_deltanet": {
    "id": "hgrn2_gated_deltanet",
    "name": "HGRN-2 Gated DeltaNet Hybrid Attention Module.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "forget_gate_visible",
      "input_gate_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "hgrn2_layer": {
    "id": "hgrn2_layer",
    "name": "Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "forget_gate_visible",
      "input_gate_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "hgrn2_outer": {
    "id": "hgrn2_outer",
    "name": "HGRN-2 Outer-Product Recurrent Linear Attention Module.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "forget_gate_visible",
      "input_gate_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "hla_attention": {
    "id": "hla_attention",
    "name": "Hadamard Linear Attention (HLA).",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "hla_inject": {
    "id": "hla_inject",
    "name": "Deprecated stub. Re-exports base module for registry compatibilit",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "hymba_transx": {
    "id": "hymba_transx",
    "name": "HymbaTransX: Hybrid Differential Transformer & State Space Model ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "infini_attention": {
    "id": "infini_attention",
    "name": "Infini-Attention with Compressive Memory and Sliding Window (Prod",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "jamba": {
    "id": "jamba",
    "name": "Jamba-2 Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block",
    "category": "I",
    "subcategory": "hybrid_composite",
    "states": [
      "idle",
      "layer_route",
      "hybrid_ssm_step",
      "hybrid_attn_step",
      "fusion_gate",
      "output"
    ],
    "substates": [
      "layer_index_indicator_visible"
    ],
    "defaultState": "hybrid_ssm_step",
    "forbidden": [
      "pure_dense_only",
      "svd_singular_values"
    ]
  },
  "jamba_info": {
    "id": "jamba_info",
    "name": "Information-Bottlenecked Jamba Architecture with Dual Operational",
    "category": "I",
    "subcategory": "hybrid_composite",
    "states": [
      "idle",
      "layer_route",
      "hybrid_ssm_step",
      "hybrid_attn_step",
      "fusion_gate",
      "output"
    ],
    "substates": [
      "layer_index_indicator_visible"
    ],
    "defaultState": "hybrid_ssm_step",
    "forbidden": [
      "pure_dense_only",
      "svd_singular_values"
    ]
  },
  "kda_attention": {
    "id": "kda_attention",
    "name": "Kernelized Decay Attention (KDA) / Recurrent Linear Attention (Ha",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "kda_inject": {
    "id": "kda_inject",
    "name": "Deprecated stub. Re-exports base module for registry compatibilit",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "leka_v1": {
    "id": "leka_v1",
    "name": "LEKAv1: Linear-time Efficient Kernel Attention (Production-Harden",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "leka_v1_inject": {
    "id": "leka_v1_inject",
    "name": "Deprecated stub. Re-exports base module for registry compatibilit",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "leka_v2": {
    "id": "leka_v2",
    "name": "LEKAv2: Linear-time Efficient Kernel Attention with Landmark Dist",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "leka_v2_inject": {
    "id": "leka_v2_inject",
    "name": "LEKA v2: Landmark-Enhanced Kernel Attention with Dual-Scale Disti",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "leka_v3": {
    "id": "leka_v3",
    "name": "LEKAv3: Linear-time Efficient Kernel Attention with Value-Scored ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "leka_v3_inject": {
    "id": "leka_v3_inject",
    "name": "LEKA v3: Value-Aware Landmark Kernel Attention with Padding Invar",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "linformer_attention": {
    "id": "linformer_attention",
    "name": "Linformer Attention: Linear-Complexity Sequence Projection via Lo",
    "category": "B",
    "subcategory": "linformer",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [
      "low_rank_projection_visible"
    ],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "lola_attention": {
    "id": "lola_attention",
    "name": "Lola Attention: Low-Rank Localized Attention via Three-Memory Sys",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "lora_attention": {
    "id": "lora_attention",
    "name": "LoRA Attention: Parameter-Efficient Low-Rank Adaptation",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "lsh_attention": {
    "id": "lsh_attention",
    "name": "Locality-Sensitive Hashing Attention: Angular Random Hyperplane B",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "mamba": {
    "id": "mamba",
    "name": "Mamba: Gated State Space Model (Mamba) linear attention replaceme",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "mamba_inject": {
    "id": "mamba_inject",
    "name": "Mamba Injection Module.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "mamba_svd_flash": {
    "id": "mamba_svd_flash",
    "name": "MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention Module (P",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "minimal_aurora": {
    "id": "minimal_aurora",
    "name": "MinimalAURORA Attention: Energy-Preserving SVD-Factorized Attenti",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "minimal_nova": {
    "id": "minimal_nova",
    "name": "MinimalNOVA Attention: SVD-Factorized Low-Rank Projection Attenti",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "minimal_prism": {
    "id": "minimal_prism",
    "name": "MinimalPRISM Attention: Singular Spectrum Energy-Factorized Low-R",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "minimal_svd": {
    "id": "minimal_svd",
    "name": "MinimalSVD: Cumulative Energy-Budgeted SVD Factorization Attentio",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "minimal_svd_v2": {
    "id": "minimal_svd_v2",
    "name": "MinimalSVD v2: Refined SVD Low-Rank Attention with Flexible Atten",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "mixattention_moska": {
    "id": "mixattention_moska",
    "name": "MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FA",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "mla_attention": {
    "id": "mla_attention",
    "name": "Multi-Head Latent Attention (MLA): Low-Rank Latent Bottleneck Com",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "moba": {
    "id": "moba",
    "name": "MoBA: Mixture of Block Attention for Long-Context Modeling.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "moba_attention": {
    "id": "moba_attention",
    "name": "MoBA Attention: Mixture-of-Block Attention via Centroid Cosine Ro",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "moe_deltakv_paged": {
    "id": "moe_deltakv_paged",
    "name": "MoE-DeltaKV Paged: Quantized & Paged KV-Cache Mixture-of-Experts ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "nasc_attention": {
    "id": "nasc_attention",
    "name": "NASC: NEXUS-AEGIS Spectral Cascade Attention.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "nexus": {
    "id": "nexus",
    "name": "NEXUS: Neural Efficient X-scoring Unified Sparse Attention.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "nha_attention": {
    "id": "nha_attention",
    "name": "Native Hybrid Attention (NHA) – Production-Hardened.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "nha_compressor": {
    "id": "nha_compressor",
    "name": "Native Hybrid Attention (NHA) Compressor.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "nsa_attention": {
    "id": "nsa_attention",
    "name": "Native Sparse Attention: Hybrid Local Sliding-Window and Dynamic ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "nystrom_attention": {
    "id": "nystrom_attention",
    "name": "Nyström Attention: Low-Rank Landmark Approximation via Matrix Pse",
    "category": "B",
    "subcategory": "nystrom",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [
      "landmark_selection_visible"
    ],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "optimized_hybrid_attention": {
    "id": "optimized_hybrid_attention",
    "name": "Optimized Hybrid Attention (OHA): Fused Sliding-Window Local Self",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "orion": {
    "id": "orion",
    "name": "ORION: Optimized Rectified Input-Aware Orthogonal Network with En",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "performer_attention": {
    "id": "performer_attention",
    "name": "Performer Attention – Fast Attention via Positive Orthogonal Rand",
    "category": "B",
    "subcategory": "performer",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [
      "kernel_features_visible"
    ],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "phantom": {
    "id": "phantom",
    "name": "PHANTOM: Parameter-Hashed AttentioN with Token-wise Orthogonal Ma",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "primal_attention": {
    "id": "primal_attention",
    "name": "Primal Attention: Self-Attention through Asymmetric Kernel SVD in",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "progressive_token_pruning": {
    "id": "progressive_token_pruning",
    "name": "Progressive Token Pruning (PTP) – Production-Hardened.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "projection_factorized_attn": {
    "id": "projection_factorized_attn",
    "name": "Projection Factorized SVD Linear Attention.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "qcqa_attention": {
    "id": "qcqa_attention",
    "name": "Quality and Capacity-Aware Grouped Query Attention (QCQA): Dynami",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "rala_attention": {
    "id": "rala_attention",
    "name": "Rank-Augmented Linear Attention (RALA): Low-Rank Parametric Head ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "rala_compressor": {
    "id": "rala_compressor",
    "name": "RALA Compressor: Rank-Augmented Linear Attention Layer Architectu",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "reap_moe": {
    "id": "reap_moe",
    "name": "REAP-MoE: Robust Expert Allocation and Pruning Mixture of Experts",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "regla_attention": {
    "id": "regla_attention",
    "name": "Refined Gated Linear Attention (ReGLA): Element-Wise Key Gating &",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "replace": {
    "id": "replace",
    "name": "Model Attention Replacement and Dynamic Graph Surgery Utilities.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "res_svd": {
    "id": "res_svd",
    "name": "Spectral Residual Low-Rank SVD Attention (Res-SVD Attention) – Pr",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "routing_attention": {
    "id": "routing_attention",
    "name": "Routing Attention: Content-Adaptive Query-Centric Clustering via ",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "rpc": {
    "id": "rpc",
    "name": "Random Projection Compressed (RPC) Attention.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "rpc_attention": {
    "id": "rpc_attention",
    "name": "RPC Attention: Randomized Phase Cluster Attention via Fixed Rando",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "salt": {
    "id": "salt",
    "name": "SALT Attention: Singular Value Adaptation with Low-Rank Tuning.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "salt_attention": {
    "id": "salt_attention",
    "name": "SALT Attention: Singular Value Adaptation with Low-Rank Tuning fo",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "samba_m3": {
    "id": "samba_m3",
    "name": "SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Bloc",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "samba_moe": {
    "id": "samba_moe",
    "name": "Samba-MoE: Selective State Space (SAMBA-M3) + REAP-MoE Hybrid Arc",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "simple_ssm": {
    "id": "simple_ssm",
    "name": "Simple State Space Model (SimpleSSM).",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "smart_weighted_ensemble": {
    "id": "smart_weighted_ensemble",
    "name": "Smart Weighted Ensemble: Task-Aware Attention & Dynamic Confidenc",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "soa_attn": {
    "id": "soa_attn",
    "name": "Shared Orthogonal Attention (SOA) with Per-Projection Latent Bott",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "soa_fused_module": {
    "id": "soa_fused_module",
    "name": "SOA-Fused Attention: Low-Rank Latent Space Multi-Head Attention w",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "soa_module": {
    "id": "soa_module",
    "name": "SOA Attention: Shared Orthogonal Attention with Latent Bottleneck",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "span_expanded_attention": {
    "id": "span_expanded_attention",
    "name": "Span-Expanded Attention: Receptive Field Extension with Parallel ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "spectra_v2": {
    "id": "spectra_v2",
    "name": "SPECTRA-v2: Sparse Per-Head Efficient Context Transformer with Re",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "spectra_v2_attention": {
    "id": "spectra_v2_attention",
    "name": "Re-export module for SpectraV2Attention.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "spectral_mamba": {
    "id": "spectral_mamba",
    "name": "Spectral-Mamba Attention Module: Hybrid Attention-State Space Mod",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "spectral_mamba_attention": {
    "id": "spectral_mamba_attention",
    "name": "Re-export module for SpectralMambaAttention.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "spectron": {
    "id": "spectron",
    "name": "SpectronAttention: Importance-Aware Low-Rank + Sparse Attention.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "spectron_attention": {
    "id": "spectron_attention",
    "name": "Spectron Attention Re-export Module.",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "squeezing_heads": {
    "id": "squeezing_heads",
    "name": "Squeezing Heads Attention Re-export Module.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "squeezing_heads_distill": {
    "id": "squeezing_heads_distill",
    "name": "Squeezing Heads Distillation: Multi-Head Receptive Field Distilla",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "squeezing_heads_distiller": {
    "id": "squeezing_heads_distiller",
    "name": "Squeezing Heads Distiller: Progressive Multi-Head Knowledge Disti",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "svd_attention": {
    "id": "svd_attention",
    "name": "SVD Attention: Sequence-Dimension Singular Subspace Projection Ar",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_compressed": {
    "id": "svd_compressed",
    "name": "SVD-Compressed Attention: Importance-Aware Lagrangian Rank Alloca",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_compressed_attention": {
    "id": "svd_compressed_attention",
    "name": "SVD-Compressed Attention Re-export Module.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_linear_attention": {
    "id": "svd_linear_attention",
    "name": "SVD Factorized Linear Attention & Kernelized Linear Attention Mod",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "svd_linear_decay": {
    "id": "svd_linear_decay",
    "name": "SVD Linear Decay Gated Attention Alias.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "svd_llm_v2": {
    "id": "svd_llm_v2",
    "name": "SVD-LLM V2 Attention Alias.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_llm_v2_attention": {
    "id": "svd_llm_v2_attention",
    "name": "SVD-LLM V2 Attention: Truncation-Aware Explicit Factor Buffer Arc",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_mamba": {
    "id": "svd_mamba",
    "name": "SVD-Mamba Attention: Hardware-Efficient State-Space Recurrence vi",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "svd_proj_attn": {
    "id": "svd_proj_attn",
    "name": "SVD Projection Attention Alias.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_projection_attention": {
    "id": "svd_projection_attention",
    "name": "SVD Projection Attention: Explicit Low-Rank Matrix Reconstruction",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_sequence_attention": {
    "id": "svd_sequence_attention",
    "name": "Low-Rank SVD Sequence Attention Re-export Module.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "template_attention": {
    "id": "template_attention",
    "name": "Template Attention: Reference Implementation & Standardization Ba",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "token_compressed": {
    "id": "token_compressed",
    "name": "Token-Compressed Attention Wrapper: Structural Abstraction & Pass",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "token_compressed_attention": {
    "id": "token_compressed_attention",
    "name": "Token Compressed Attention: Saliency-Based Key-Value Pruning",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "tpa_attention": {
    "id": "tpa_attention",
    "name": "TPAttention: Tensor Product Attention with Factorised Query-Key P",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "tt": {
    "id": "tt",
    "name": "TokenCompressedAttention: A No‑Compression Baseline Wrapper for A",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "unified_rope": {
    "id": "unified_rope",
    "name": "Universal Rotary Position Embedding (UnifiedRoPE): Positional Inv",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "upi_mamba": {
    "id": "upi_mamba",
    "name": "UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "upi_stuffed_mamba": {
    "id": "upi_stuffed_mamba",
    "name": "UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM B",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "vig_attention": {
    "id": "vig_attention",
    "name": "ViG Attention: Vision GNN-inspired 2D Grid Linear Attention with ",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "vspec": {
    "id": "vspec",
    "name": "VSpec++ Attention: Variational Spectral Token-Adaptive Attention ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "weighted_ensemble": {
    "id": "weighted_ensemble",
    "name": "Smart Weighted Ensemble with Task-Aware Attention and Hybrid Comb",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "wgqa_attention": {
    "id": "wgqa_attention",
    "name": "Weighted Grouped-Query Attention (WGQA).",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [
      "kv_sharing_visible"
    ],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "xattention": {
    "id": "xattention",
    "name": "XAttention: Block-Sparse Coarse-to-Fine Routing with Unified Soft",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "zeros_attention": {
    "id": "zeros_attention",
    "name": "ZeroS Attention: Zero-Mean Normalized Linear Attention",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "a3_compressor": {
    "id": "a3_compressor",
    "name": "A3 Compressor: Activation-Aware Factorized SVD Compression with F",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "fisher_weights_visible",
      "ln_recalibration_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aa_svd_compressor": {
    "id": "aa_svd_compressor",
    "name": "Activation-Aware Singular Value Decomposition (AASVD) Compressor.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "aasvd": {
    "id": "aasvd",
    "name": "Activation-Aware Singular Value Decomposition (AASVD): Streaming ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "adaptive_compressor": {
    "id": "adaptive_compressor",
    "name": "AdaptiveTransformerCompressor: Dynamic Energy-Threshold Attention",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "adaptive_ranks": {
    "id": "adaptive_ranks",
    "name": "Adaptive Ranks SVD Compressor: dynamic rank selection per layer b",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "adasvd_compressor": {
    "id": "adasvd_compressor",
    "name": "AdaSVDCompressor: Adaptive SVD via Nuclear-Norm Layer Importance ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "admm_compressor": {
    "id": "admm_compressor",
    "name": "ADMM Compressor: Alternating Direction Method of Multipliers for ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "admm_pruner": {
    "id": "admm_pruner",
    "name": "ADMM Pruner: Alternating Direction Method of Multipliers with Dyn",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "armor_pruner": {
    "id": "armor_pruner",
    "name": "ARMOR Pruner: Adaptive Resilient Matrix Orthogonal Rank Pruning (",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "ast_pruner": {
    "id": "ast_pruner",
    "name": "AST: Adaptive Semi-Structured 2:4 and N:M Pruning with Empirical ",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "astro_quantizer": {
    "id": "astro_quantizer",
    "name": "AstroQuantizer: Dynamic Uniform Range Quantization Engine with St",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "bonsai_pruner": {
    "id": "bonsai_pruner",
    "name": "Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradien",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "cachenotes_compressor": {
    "id": "cachenotes_compressor",
    "name": "CacheNotes Compressor: Task-Aware CPT Guidance for Static SVD & D",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "come": {
    "id": "come",
    "name": "CoMe: Compress LLMs through Layer Concatenation and Shedding with",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "compot_compressor": {
    "id": "compot_compressor",
    "name": "COMPOT: Compressed Orthogonal Dictionary Learning Compressor.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "coopq_quantizer": {
    "id": "coopq_quantizer",
    "name": "CoopQ: Cooperative Game-Theoretic Bit Allocation Quantizer with S",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "cr_compressor": {
    "id": "cr_compressor",
    "name": "CR Compressor: Interpolative Decomposition for Weight Compression",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "cur_compressor": {
    "id": "cur_compressor",
    "name": "CUR Compressor: replaces Linear and Conv1D layers with CUR decomp",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "cur_linear": {
    "id": "cur_linear",
    "name": "CUR Linear Layer: low-rank cross-approximation using column and r",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "cur_moe": {
    "id": "cur_moe",
    "name": "CUR-MoE Compressor alias module.",
    "category": "F",
    "subcategory": "mixture_of_experts",
    "states": [
      "idle",
      "router_gate",
      "expert_dispatch",
      "expert_compute",
      "expert_combine",
      "output"
    ],
    "substates": [
      "routing_probabilities_visible"
    ],
    "defaultState": "router_gate",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "cur_moe_compressor": {
    "id": "cur_moe_compressor",
    "name": "CUR-MoE Compressor: apply CUR cross-approximation to Mixture of E",
    "category": "F",
    "subcategory": "mixture_of_experts",
    "states": [
      "idle",
      "router_gate",
      "expert_dispatch",
      "expert_compute",
      "expert_combine",
      "output"
    ],
    "substates": [
      "routing_probabilities_visible"
    ],
    "defaultState": "router_gate",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "cure": {
    "id": "cure",
    "name": "CURE: CUR Expert Compression Engine with Randomized Subspace Appr",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "delta_kv": {
    "id": "delta_kv",
    "name": "DeltaKV: Residual-Based Key-Value Cache Compression via Long-Rang",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "direct_svd": {
    "id": "direct_svd",
    "name": "Direct SVD Compressor: Truncated SVD low-rank factorized layer re",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "dobi": {
    "id": "dobi",
    "name": "DobiSVDLinearLayer & Compressor: Differentiable SVD with Learnabl",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "dobi_svd": {
    "id": "dobi_svd",
    "name": "DobiSVDLinearLayer & Compressor: Differentiable SVD with Learnabl",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "dynamic_entropy": {
    "id": "dynamic_entropy",
    "name": "Dynamic Entropy (`dynamic_entropy`)",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "dynamic_rank": {
    "id": "dynamic_rank",
    "name": "Dynamic Entropy Singular Value Decomposition (Dynamic Entropy SVD",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "enlighten_compressor": {
    "id": "enlighten_compressor",
    "name": "ENLighten: Low-Rank SVD + 2:4 Structured Sparsity (Photonic-Align",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "entropy_budget_allocation": {
    "id": "entropy_budget_allocation",
    "name": "EntropyBudgetAllocation: global rank assignment via whitened acti",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "erc_svd": {
    "id": "erc_svd",
    "name": "ERCCompressor: Error-Controlled Singular Value Energy Thresholdin",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "faq_quantizer": {
    "id": "faq_quantizer",
    "name": "FAQ (Feature-Activation-Aware Quantization): adjusts quantization",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "flap": {
    "id": "flap",
    "name": "FLAP: Fluctuation-Based Adaptive Structured Pruning.",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "flat_llm": {
    "id": "flat_llm",
    "name": "FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning ",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "frandi": {
    "id": "frandi",
    "name": "FRanDI Compressor alias module.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "frandi_compressor": {
    "id": "frandi_compressor",
    "name": "FRanDI: Feature-Regression Aligned Singular Value Decomposition C",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "gfws_d_compressor": {
    "id": "gfws_d_compressor",
    "name": "GFWSVD: Fisher-weighted / Covariance-weighted SVD. Registered as ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "grecodec": {
    "id": "grecodec",
    "name": "GreCoDeC Compressor alias module.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "grecodec_compressor": {
    "id": "grecodec_compressor",
    "name": "GreCoDeC: Greedy Coordinate Descent Low-Rank Compressor.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "hashed_linear": {
    "id": "hashed_linear",
    "name": "HashedNets: Vectorized Weight Sharing via Hash Tables with Sign H",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "hashed_net_compressor": {
    "id": "hashed_net_compressor",
    "name": "HashedNet Compressor: replaces nn.Linear and Conv1D layers with H",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "huffman_coder": {
    "id": "huffman_coder",
    "name": "Huffman Coder & Entropy Compression Engine for Weight and KV Cach",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "hybrid_svd_layer": {
    "id": "hybrid_svd_layer",
    "name": "Hybrid SVD Layer: Single‑Head SVD Approximation with Positional E",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "hybridkv": {
    "id": "hybridkv",
    "name": "HybridKV: Hybrid KV Cache Compression for Multimodal & Long-Conte",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "hybridkv_compressor": {
    "id": "hybridkv_compressor",
    "name": "HybridKV Compressor: Dual Static-Dynamic Head-Specific Hierarchic",
    "category": "I",
    "subcategory": "hybrid_composite",
    "states": [
      "idle",
      "layer_route",
      "hybrid_ssm_step",
      "hybrid_attn_step",
      "fusion_gate",
      "output"
    ],
    "substates": [
      "layer_index_indicator_visible"
    ],
    "defaultState": "hybrid_ssm_step",
    "forbidden": [
      "pure_dense_only",
      "svd_singular_values"
    ]
  },
  "kmeans_quantize": {
    "id": "kmeans_quantize",
    "name": "KMeans-based weight clustering quantizer with STE. Registered as ",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "kronecker_compressor": {
    "id": "kronecker_compressor",
    "name": "Kronecker Product Compressor: replaces nn.Linear and Conv1D layer",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "kronecker_linear": {
    "id": "kronecker_linear",
    "name": "Kronecker Linear Layer: weight matrix represented as Kronecker pr",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "kronecker_tt_lora": {
    "id": "kronecker_tt_lora",
    "name": "Kronecker-TT-LoRA: Hybrid Compression Engine for Parameter-Effici",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "kvtc": {
    "id": "kvtc",
    "name": "KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive qu",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "kvtc_compressor": {
    "id": "kvtc_compressor",
    "name": "KVTC Compressor: Key-Value Transform Coding via PCA Decorrelation",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "liftr": {
    "id": "liftr",
    "name": "LIFTR: Low-Rank In-Flight Truncated Reconstruction Compressor",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "lillama_compressor": {
    "id": "lillama_compressor",
    "name": "LiLLaMA: Local Distillation via Activation Cross-Covariance Singu",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "lsc_compressor": {
    "id": "lsc_compressor",
    "name": "LSC Compressor: Low-Rank Spectral Fourier Compression Engine",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "mamba_shedder": {
    "id": "mamba_shedder",
    "name": "Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer She",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "state_space_visible",
      "selective_scan_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ]
  },
  "mgie_svd": {
    "id": "mgie_svd",
    "name": "MGIE-SVD: Maximum Information Entropy SVD Rank Selection.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "mgie_svd_compressor": {
    "id": "mgie_svd_compressor",
    "name": "MGIESVDBudgetCompressor: Information Entropy SVD Rank Allocation ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "mlorq_compressor": {
    "id": "mlorq_compressor",
    "name": "MLoRQ Compressor: Multi-Layer Optimal Rank SVD and Joint Low-Bit ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "modegpt": {
    "id": "modegpt",
    "name": "MoDeGPT: Modular Decomposition of Transformer Blocks.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "modegpt_compressor": {
    "id": "modegpt_compressor",
    "name": "MoDeGPT: Modular Decomposition of Transformer Blocks.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "nli_softmax": {
    "id": "nli_softmax",
    "name": "NLI Softmax approximation – a fast, quantisation‑friendly softmax",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "nmf_compressor": {
    "id": "nmf_compressor",
    "name": "NMFCompressor: Non-Negative Matrix Factorization Compression Engi",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "oats": {
    "id": "oats",
    "name": "OATS: Outlier-Aware Training Sparsification. Registered as 'oats'",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "oats_compressor": {
    "id": "oats_compressor",
    "name": "OATS: Outlier-Aware Training & Saliency-Based Sparsification with",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "optimal_subspace_projection": {
    "id": "optimal_subspace_projection",
    "name": "Optimal Subspace Projection: Activation-Aware Subspace Projection",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "padre_attention": {
    "id": "padre_attention",
    "name": "PADRe: Polynomial Attention with Dropout Regularization.",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "proxsparse": {
    "id": "proxsparse",
    "name": "ProxSparse: Proximal Gradient Regularized Structured & Semi-Struc",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "proxsparse_pruner": {
    "id": "proxsparse_pruner",
    "name": "ProxSparse: regularized learning of semi-structured sparsity. Reg",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "pruning_utils": {
    "id": "pruning_utils",
    "name": "Pruning Utilities and Saliency Estimation for Structural Pruning ",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "threshold_cutoff_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "replaceme_compressor": {
    "id": "replaceme_compressor",
    "name": "ReplaceMe: Layer Removal and Replacement with Linear Residual Blo",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "roast_hashed_linear": {
    "id": "roast_hashed_linear",
    "name": "ROAST: Hardware-Aware Weight Sharing with Tiling.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "rotational_slicegpt_int4": {
    "id": "rotational_slicegpt_int4",
    "name": "Rotational SliceGPT INT4 Compressor: Rotational Invariance, Subsp",
    "category": "H",
    "subcategory": "quantization",
    "states": [
      "idle",
      "calibration",
      "scale_calc",
      "quantize_int8_fp8",
      "dequantize",
      "output"
    ],
    "substates": [
      "smoothing_factors_visible"
    ],
    "defaultState": "quantize_int8_fp8",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "rpca": {
    "id": "rpca",
    "name": "RPCA Compressor alias module.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "rpca_compressor": {
    "id": "rpca_compressor",
    "name": "RPCACompressor: Robust Principal Component Analysis Low-Rank & Sp",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "saes_svd": {
    "id": "saes_svd",
    "name": "SAES-SVD v2: error suppression with fixed rank ratio. Registered ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "saes_svd_compressor": {
    "id": "saes_svd_compressor",
    "name": "SAES-SVD Compressor: Joint intra- and inter-layer error compensat",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "saes_svd_v2": {
    "id": "saes_svd_v2",
    "name": "SAES-SVD v2: Spectrum-Aware Error-Suppressed Singular Value Decom",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "salr": {
    "id": "salr",
    "name": "SALR: Structure-Aware / Sparse Low-Rank compression combining SVD",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "sdrm": {
    "id": "sdrm",
    "name": "SDRM: Scaled Dimension Rank Modulation Compressor. Registered as ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "sdrm_compressor": {
    "id": "sdrm_compressor",
    "name": "SDRM: Rank Assignment Proportional to Parameter Budget for Edge I",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "slicegpt": {
    "id": "slicegpt",
    "name": "SliceGPT: Post-Training Slicing with Orthogonal Transformation & ",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "slicegpt_compressor": {
    "id": "slicegpt_compressor",
    "name": "SliceGPT PCA-based slicing compressor. Registered as 'slicegpt_pc",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "slicegpt_pca": {
    "id": "slicegpt_pca",
    "name": "SliceGPT PCA Compressor: Principal Component Analysis based Dimen",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "slim_compressor": {
    "id": "slim_compressor",
    "name": "SLiM: Structured Low-Rank Integrated Matrix Compression (Quantiza",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "soka": {
    "id": "soka",
    "name": "SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor. Re",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "soka_compressor": {
    "id": "soka_compressor",
    "name": "SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "sparsegpt": {
    "id": "sparsegpt",
    "name": "SparseGPT: One-Shot Second-Order Hessian-Guided Weight Pruning fo",
    "category": "G",
    "subcategory": "pruning_sparsity",
    "states": [
      "idle",
      "importance_eval",
      "mask_generation",
      "pruned_projection",
      "output"
    ],
    "substates": [
      "hessian_update_visible"
    ],
    "defaultState": "importance_eval",
    "forbidden": [
      "dense_attention_matrix",
      "svd_singular_values",
      "softmax"
    ]
  },
  "spectraformer_attention": {
    "id": "spectraformer_attention",
    "name": "Spectraformer: Random Feature Linear Attention with Flexible Kern",
    "category": "B",
    "subcategory": "sparse_approximate",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ]
  },
  "subspace_projection": {
    "id": "subspace_projection",
    "name": "Subspace Projection Compressor: Low-Rank Subspace Projection and ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "svd_linear": {
    "id": "svd_linear",
    "name": "SVD-based low-rank linear layer replacement with optional Fused S",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "swift_svd": {
    "id": "swift_svd",
    "name": "Swift-SVD: fast closed-form truncated SVD with symmetric square-r",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "swift_svd_act": {
    "id": "swift_svd_act",
    "name": "Swift-SVD Activation-Aware Compressor via cross-covariance / acti",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "swift_svd_compressor": {
    "id": "swift_svd_compressor",
    "name": "Swift-SVD Activation-aware via cross-covariance. Registered as 's",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "tlinformer_attention": {
    "id": "tlinformer_attention",
    "name": "TLinFormer: Exact linear‑time attention via low‑rank sequence pro",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "tomoe": {
    "id": "tomoe",
    "name": "ToMoE: Converting Dense MLP into Mixture of Experts via Dynamic S",
    "category": "F",
    "subcategory": "mixture_of_experts",
    "states": [
      "idle",
      "router_gate",
      "expert_dispatch",
      "expert_compute",
      "expert_combine",
      "output"
    ],
    "substates": [
      "routing_probabilities_visible"
    ],
    "defaultState": "router_gate",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "tomoe_compressor": {
    "id": "tomoe_compressor",
    "name": "ToMoE: converts dense linear and Conv1D layers (e.g. FFN and atte",
    "category": "F",
    "subcategory": "mixture_of_experts",
    "states": [
      "idle",
      "router_gate",
      "expert_dispatch",
      "expert_compute",
      "expert_combine",
      "output"
    ],
    "substates": [
      "routing_probabilities_visible"
    ],
    "defaultState": "router_gate",
    "forbidden": [
      "svd_singular_values",
      "state_space_scan",
      "quadratic_square"
    ]
  },
  "tt_compressor": {
    "id": "tt_compressor",
    "name": "Tensor Train Compressor: replaces eligible nn.Linear and Conv1D l",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "tt_linear": {
    "id": "tt_linear",
    "name": "Tensor-Train Linear Layer: parameterizes weight matrices as Tenso",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "uniql_compressor": {
    "id": "uniql_compressor",
    "name": "UniQL: Joint low‑rank + quantisation using SVD and uniform quant.",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ]
  },
  "zerokv_neo": {
    "id": "zerokv_neo",
    "name": "ZeroKV-Neo: L2-norm based KV cache eviction with attention sink a",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "zerokv_neo_compressor": {
    "id": "zerokv_neo_compressor",
    "name": "ZeroKV-Neo: L2-norm based KV cache eviction with attention sink a",
    "category": "A",
    "subcategory": "dense_attention",
    "states": [
      "idle",
      "qkv_projection",
      "score_matrix",
      "softmax",
      "context",
      "output"
    ],
    "substates": [],
    "defaultState": "qkv_projection",
    "forbidden": [
      "state_space_scan",
      "svd_singular_values",
      "recurrent_hidden_state"
    ]
  },
  "a3": {
    "id": "a3",
    "name": "A3 Compressor: Activation-Aware Factorized SVD Compression with F",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "fisher_weights_visible",
      "ln_recalibration_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ],
    "targetId": "a3_compressor"
  },
  "hgrn2": {
    "id": "hgrn2",
    "name": "HGRN-2 Outer-Product Recurrent Linear Attention Module.",
    "category": "D",
    "subcategory": "recurrent_linear",
    "states": [
      "idle",
      "t0",
      "t1",
      "t2",
      "t3",
      "t4",
      "output"
    ],
    "substates": [
      "forget_gate_visible",
      "input_gate_visible"
    ],
    "defaultState": "t1",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "square_attention",
      "dense_score_matrix"
    ],
    "targetId": "hgrn2_outer"
  },
  "performer": {
    "id": "performer",
    "name": "Performer Attention – Fast Attention via Positive Orthogonal Rand",
    "category": "B",
    "subcategory": "performer",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [
      "kernel_features_visible"
    ],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ],
    "targetId": "performer_attention"
  },
  "linformer": {
    "id": "linformer",
    "name": "Linformer Attention: Linear-Complexity Sequence Projection via Lo",
    "category": "B",
    "subcategory": "linformer",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [
      "low_rank_projection_visible"
    ],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ],
    "targetId": "linformer_attention"
  },
  "nystrom": {
    "id": "nystrom",
    "name": "Nyström Attention: Low-Rank Landmark Approximation via Matrix Pse",
    "category": "B",
    "subcategory": "nystrom",
    "states": [
      "idle",
      "feature_map",
      "associative_reorder",
      "output"
    ],
    "substates": [
      "landmark_selection_visible"
    ],
    "defaultState": "feature_map",
    "forbidden": [
      "dense_attention_matrix",
      "full_softmax_matrix",
      "quadratic_square"
    ],
    "targetId": "nystrom_attention"
  },
  "asvd": {
    "id": "asvd",
    "name": "ASVD Attention: Activation-Aware Singular Value Decomposition Arc",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "no_weighting"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ],
    "targetId": "asvd_attention"
  },
  "adaptive_svd": {
    "id": "adaptive_svd",
    "name": "Adaptive SVD Attention: Low-Rank SVD Projections with Parallel Re",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ],
    "targetId": "adaptive_svd_attention"
  },
  "adasvd": {
    "id": "adasvd",
    "name": "AdaSVDCompressor: Adaptive SVD via Nuclear-Norm Layer Importance ",
    "category": "E",
    "subcategory": "svd_low_rank",
    "states": [
      "idle",
      "show_W",
      "decompose",
      "singular_values",
      "truncate",
      "reconstruct",
      "error_heatmap"
    ],
    "substates": [
      "energy_threshold_visible"
    ],
    "defaultState": "decompose",
    "forbidden": [
      "qkv",
      "softmax",
      "attention_matrix",
      "causal_mask",
      "square_attention"
    ],
    "targetId": "adasvd_compressor"
  }
};

/**
 * Validate that a module state is legal and never violates forbidden constraints.
 * Throws an Error if a forbidden state or undeclared state is rendered.
 */
export function validateModuleState(moduleId, state, substate = null) {
  const config = ROUTING_MATRIX[moduleId];
  if (!config) {
    throw new Error(`Unknown module "${moduleId}" in ROUTING_MATRIX`);
  }

  // Check forbidden states
  if (config.forbidden && config.forbidden.includes(state)) {
    throw new Error(`FORBIDDEN STATE VIOLATION: Module "${moduleId}" (Category ${config.category}) strictly forbids state "${state}"!`);
  }

  // Check substate against forbidden
  if (substate && config.forbidden && config.forbidden.includes(substate)) {
    throw new Error(`FORBIDDEN SUBSTATE VIOLATION: Module "${moduleId}" (Category ${config.category}) strictly forbids substate "${substate}"!`);
  }

  // Category D strict invariant: NO QKV, Softmax, Attention Matrix, Dense Score Matrix
  if (config.category === "D") {
    const forbiddenPatterns = ["qkv", "softmax", "attention_matrix", "square_attention", "dense_score_matrix"];
    if (forbiddenPatterns.some(p => (state || "").toLowerCase().includes(p))) {
      throw new Error(`CATEGORY D FORBIDDEN: Recurrent module "${moduleId}" attempted to render S×S attention/softmax state "${state}"!`);
    }
  }

  // Category E strict invariant: NO QKV, Softmax, Attention Matrix, Causal Mask
  if (config.category === "E") {
    const forbiddenPatterns = ["qkv", "softmax", "attention_matrix", "causal_mask"];
    if (forbiddenPatterns.some(p => (state || "").toLowerCase().includes(p))) {
      throw new Error(`CATEGORY E FORBIDDEN: SVD module "${moduleId}" attempted to render attention/softmax state "${state}"!`);
    }
  }

  // Category B strict invariant: NO Dense Attention Matrix, Full Softmax Matrix
  if (config.category === "B") {
    const forbiddenPatterns = ["dense_attention_matrix", "full_softmax_matrix", "quadratic_square"];
    if (forbiddenPatterns.some(p => (state || "").toLowerCase().includes(p))) {
      throw new Error(`CATEGORY B FORBIDDEN: Sparse module "${moduleId}" attempted to render dense quadratic matrix "${state}"!`);
    }
  }

  if (state && !config.states.includes(state)) {
    throw new Error(`Invalid state "${state}" for module "${moduleId}". Valid states: ${config.states.join(", ")}`);
  }

  return true;
}

export function getModuleRouting(moduleId) {
  return ROUTING_MATRIX[moduleId] || null;
}

export default ROUTING_MATRIX;
