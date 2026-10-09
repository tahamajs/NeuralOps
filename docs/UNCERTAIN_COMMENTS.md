# UNCERTAIN Comments Table (Part 3.1)

Total items: 190

| # | File | Line | Comment | Why Uncertain | Proposed Action |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | `neuralops/core/_attention_extractors.py` | 70 | `# Also try nested attention.self.query` | Ambiguous comment requiring manual confirmation | KEEP |
| 2 | `neuralops/core/attention_mask.py` | 48 | `# True = keep, False = mask` | Ambiguous comment requiring manual confirmation | KEEP |
| 3 | `neuralops/core/rope.py` | 123 | `# In-place/fused expression: q_rot = q * cos + rotate_half(q) * sin` | Ambiguous comment requiring manual confirmation | KEEP |
| 4 | `neuralops/core/selective_scan.py` | 169 | `# SC-FIX-04: exp(-inf) = 0 exactly; no nan_to_num needed.` | Ambiguous comment requiring manual confirmation | KEEP |
| 5 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 522 | `# Output dense from outer container` | Ambiguous comment requiring manual confirmation | KEEP |
| 6 | `neuralops/attentions/gated/gated_delta_net.py` | 834 | `# z_proj from q` | Ambiguous comment requiring manual confirmation | KEEP |
| 7 | `neuralops/attentions/hybrid/fusion_attn.py` | 912 | `# take top-(k-1) from scores excluding block 0` | Ambiguous comment requiring manual confirmation | KEEP |
| 8 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 177 | `# Shape alignment (unchanged from original)` | Ambiguous comment requiring manual confirmation | KEEP |
| 9 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 215 | `# Base SVD weights are already frozen via requires_grad=False on construction.` | Ambiguous comment requiring manual confirmation | KEEP |
| 10 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 641 | `# 1) Raw validity from padding (before any causal logic)` | Ambiguous comment requiring manual confirmation | KEEP |
| 11 | `neuralops/attentions/hybrid/spectron_fp8_attention.py` | 844 | `# Init svd_U_q / svd_U_k from Q / K weights.` | Ambiguous comment requiring manual confirmation | KEEP |
| 12 | `neuralops/attentions/linear/cosformer.py` | 406 | `# Zero contribution from padded keys.` | Ambiguous comment requiring manual confirmation | KEEP |
| 13 | `neuralops/attentions/linear/nystrom_attention.py` | 630 | `# Landmarks from all previous chunks (0..c_idx-1)` | Ambiguous comment requiring manual confirmation | KEEP |
| 14 | `neuralops/attentions/linear/rala_attention.py` | 644 | `# RALA-FIX-04: removed dead scale_factor=1.0` | Ambiguous comment requiring manual confirmation | KEEP |
| 15 | `neuralops/attentions/linear/spectron.py` | 395 | `# Whitening factors per projection (from calibration)` | Ambiguous comment requiring manual confirmation | KEEP |
| 16 | `neuralops/attentions/linear/triton_cosformer.py` | 647 | `# BUG 5 FIX: fixed max_length from config` | Ambiguous comment requiring manual confirmation | KEEP |
| 17 | `neuralops/attentions/modern/nha_attention.py` | 806 | `# NHA-FIX-02: do not silently return None` | Ambiguous comment requiring manual confirmation | KEEP |
| 18 | `neuralops/attentions/sparse/cluster_attention.py` | 704 | `# Disincentivize padding tokens from pulling centroids` | Ambiguous comment requiring manual confirmation | KEEP |
| 19 | `neuralops/attentions/sparse/lsh_attention.py` | 248 | `# LSH-10: set self.config explicitly` | Ambiguous comment requiring manual confirmation | KEEP |
| 20 | `neuralops/attentions/sparse/reap_moe.py` | 1129 | `# RP-FIX-25: propagate n_experts, top_k from comp_config` | Ambiguous comment requiring manual confirmation | KEEP |
| 21 | `neuralops/attentions/ssm/hymba_transx.py` | 452 | `# HX-FIX-07: diff_lambda init = -2.0 (sigmoid ≈ 0.12)` | Ambiguous comment requiring manual confirmation | KEEP |
| 22 | `neuralops/attentions/ssm/jamba_info.py` | 389 | `# Stored as buffer (not parameter) because it's derived from lambda_init.` | Ambiguous comment requiring manual confirmation | KEEP |
| 23 | `neuralops/attentions/ssm/mamba_block.py` | 932 | `# Recompute return format (MB-12)` | Ambiguous comment requiring manual confirmation | KEEP |
| 24 | `neuralops/attentions/ssm/mamba_svd_flash.py` | 738 | `# MSF-FIX-17: initialise SSM projections from W_q, W_k using SVD` | Ambiguous comment requiring manual confirmation | KEEP |
| 25 | `neuralops/attentions/ssm/nexus.py` | 835 | `# copy_weights (unchanged from original)` | Ambiguous comment requiring manual confirmation | KEEP |
| 26 | `neuralops/attentions/ssm/selective_ssm.py` | 96 | `# Delta, B, C projections from x_ssm` | Ambiguous comment requiring manual confirmation | KEEP |
| 27 | `neuralops/attentions/ssm/selective_ssm.py` | 102 | `# A parameterization: A = -exp(A_log)` | Ambiguous comment requiring manual confirmation | KEEP |
| 28 | `neuralops/attentions/ssm/upi_stuffed_mamba.py` | 683 | `# Float mask: -inf = invalid, 0 = valid` | Ambiguous comment requiring manual confirmation | KEEP |
| 29 | `neuralops/attentions/svd/adaptive_svd_attention.py` | 705 | `# ASVD-FIX-06: always return (output, attn_weights)` | Ambiguous comment requiring manual confirmation | KEEP |
| 30 | `neuralops/attentions/svd/asvd_attention.py` | 253 | `# Factorized SVD projection: y = ((x @ V.T) * S) @ U.T + bias` | Ambiguous comment requiring manual confirmation | KEEP |
| 31 | `neuralops/attentions/svd/svd_llm_v2_attention.py` | 214 | `# Device & Dtype detection from old_attn` | Ambiguous comment requiring manual confirmation | KEEP |
| 32 | `neuralops/attentions/svd/svd_projection_attention.py` | 172 | `# Compute (x @ V^T) @ U^T = x @ (U @ V)^T` | Ambiguous comment requiring manual confirmation | KEEP |
| 33 | `neuralops/benchmarks/discovery.py` | 91 | `# Family deduction from module name` | Ambiguous comment requiring manual confirmation | KEEP |
| 34 | `neuralops/compression_engine/experimental/hashed_net_compressor.py` | 383 | `# lora_B @ lora_A / scaling = R (approx)` | Ambiguous comment requiring manual confirmation | KEEP |
| 35 | `neuralops/compression_engine/experimental/liftr.py` | 104 | `# LIFTR-FIX-05: determine return signature once at init.` | Ambiguous comment requiring manual confirmation | KEEP |
| 36 | `neuralops/compression_engine/low_rank/come_compressor.py` | 119 | `# Tuple return` | Ambiguous comment requiring manual confirmation | KEEP |
| 37 | `neuralops/compression_engine/low_rank/come_compressor.py` | 125 | `# List return` | Ambiguous comment requiring manual confirmation | KEEP |
| 38 | `neuralops/compression_engine/low_rank/come_compressor.py` | 131 | `# Dict return` | Ambiguous comment requiring manual confirmation | KEEP |
| 39 | `neuralops/compression_engine/low_rank/dora_qlora.py` | 120 | `# Magnitude parameter m = \` | \ | KEEP |
| 40 | `neuralops/compression_engine/low_rank/dora_qlora.py` | 126 | `# Effective directional weight: V = W + scaling * (B @ A)` | Ambiguous comment requiring manual confirmation | KEEP |
| 41 | `neuralops/compression_engine/low_rank/grecodec_compressor.py` | 183 | `# Update U given V: min_U \` | \ | KEEP |
| 42 | `neuralops/compression_engine/low_rank/grecodec_compressor.py` | 192 | `# Update V given U: min_V \` | \ | KEEP |
| 43 | `neuralops/compression_engine/low_rank/nmf_compressor.py` | 125 | `# (H@V * W_sign) @ x = (H@V) @ (W_sign * x) element-wise? No —` | Ambiguous comment requiring manual confirmation | KEEP |
| 44 | `neuralops/compression_engine/low_rank/oats_compressor.py` | 434 | `# Scale weights: W_scaled = W @ L_M` | Ambiguous comment requiring manual confirmation | KEEP |
| 45 | `neuralops/compression_engine/low_rank/oats_compressor.py` | 484 | `# De-scale: W_comp = (L + S_mat) @ L_M^-1` | Ambiguous comment requiring manual confirmation | KEEP |
| 46 | `neuralops/compression_engine/low_rank/salr.py` | 389 | `# Clear large tensors from GPU memory aggressively` | Ambiguous comment requiring manual confirmation | KEEP |
| 47 | `neuralops/compression_engine/pruning/armor_pruner.py` | 395 | `# AR-FIX-05: Conv1D detection via class name.` | Ambiguous comment requiring manual confirmation | KEEP |
| 48 | `neuralops/compression_engine/pruning/armor_pruner.py` | 832 | `# AR-FIX-14: distinguish storage from effective.` | Ambiguous comment requiring manual confirmation | KEEP |
| 49 | `neuralops/compression_engine/pruning/flap.py` | 359 | `# FL-FIX-01: compute actual sparsity from mask.` | Ambiguous comment requiring manual confirmation | KEEP |
| 50 | `neuralops/compression_engine/pruning/proxsparse.py` | 504 | `# Final projection (mask from last iteration already applied).` | Ambiguous comment requiring manual confirmation | KEEP |
| 51 | `neuralops/compression_engine/pruning/proxsparse.py` | 574 | `# PS-FIX-08: effective_parameter_ratio separate from compression_ratio.` | Ambiguous comment requiring manual confirmation | KEEP |
| 52 | `neuralops/compression_engine/pruning/wanda.py` | 44 | `# Compute Wanda importance metric: S = \` | \ | KEEP |
| 53 | `neuralops/compression_engine/quantization/awq.py` | 38 | `# Protect salient channels using activation magnitude: s = act^alpha` | Ambiguous comment requiring manual confirmation | KEEP |
| 54 | `neuralops/compression_engine/quantization/awq.py` | 44 | `# Restore scaling: Q = Q' / diag(s)` | Ambiguous comment requiring manual confirmation | KEEP |
| 55 | `neuralops/compression_engine/svd/aa_svd_compressor.py` | 301 | `# override from kwargs (highest priority after explicit args)` | Ambiguous comment requiring manual confirmation | KEEP |
| 56 | `neuralops/compression_engine/svd/adaptive_ranks.py` | 750 | `# Update A: A = (W @ E @ B.T) @ inv(B @ E @ B.T + eps * I)` | Ambiguous comment requiring manual confirmation | KEEP |
| 57 | `neuralops/compression_engine/svd/adaptive_ranks.py` | 762 | `# Update B: B = inv(A.T @ D @ A + eps * I) @ (A.T @ D @ W)` | Ambiguous comment requiring manual confirmation | KEEP |
| 58 | `neuralops/compression_engine/svd/dynamic_rank.py` | 81 | `# Normalized singular probability mass function p_i = sigma_i / sum(sigma_j)` | Ambiguous comment requiring manual confirmation | KEEP |
| 59 | `neuralops/compression_engine/svd/gfws_d_compressor.py` | 405 | `# Whiten weights: M = W * Cov(X)^{1/2}` | Ambiguous comment requiring manual confirmation | KEEP |
| 60 | `neuralops/compression_engine/svd/gfws_d_compressor.py` | 412 | `# De-whiten: candidate = M_r * Cov(X)^{-1/2}` | Ambiguous comment requiring manual confirmation | KEEP |
| 61 | `neuralops/compression_engine/svd/residual_svd_compressor.py` | 228 | `# Scale S_tail by adapter scaling factor gamma: (B @ A) * gamma = E_tail` | Ambiguous comment requiring manual confirmation | KEEP |
| 62 | `neuralops/compression_engine/svd/swift_svd.py` | 410 | `# SV-FIX-03: raw_shape from base_layer, correctly oriented.` | Ambiguous comment requiring manual confirmation | KEEP |
| 63 | `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py` | 721 | `# KTL-FIX-13: total stored params = base + adapter.` | Ambiguous comment requiring manual confirmation | KEEP |
| 64 | `neuralops/core/selective_scan.py` | 180 | `# Carry from previous chunk` | Ambiguous comment requiring manual confirmation | KEEP |
| 65 | `neuralops/core/triton_kernels/fms_scan.py` | 298 | `# Gradients from y_t = num_t / denom_t` | Ambiguous comment requiring manual confirmation | KEEP |
| 66 | `neuralops/core/triton_kernels/fms_scan.py` | 299 | `# dL/dnum_t = dy_t / denom_t` | Ambiguous comment requiring manual confirmation | KEEP |
| 67 | `neuralops/core/triton_kernels/fms_scan.py` | 301 | `# dL/ddenom_t = - sum(dy_t * num_t) / (denom_t * denom_t)` | Ambiguous comment requiring manual confirmation | KEEP |
| 68 | `neuralops/core/triton_kernels/fms_scan.py` | 311 | `# from num_t: sum_j S_t[i, j] * dnum_t[j]` | Ambiguous comment requiring manual confirmation | KEEP |
| 69 | `neuralops/core/triton_kernels/fms_scan.py` | 313 | `# from denom_t: Z_t * ddenom_t` | Ambiguous comment requiring manual confirmation | KEEP |
| 70 | `neuralops/core/triton_kernels/fms_scan.py` | 327 | `# from S_t: sum_j dS[i, j] * v_f[j]` | Ambiguous comment requiring manual confirmation | KEEP |
| 71 | `neuralops/core/triton_kernels/fms_scan.py` | 328 | `# from Z_t: dZ[i]` | Ambiguous comment requiring manual confirmation | KEEP |
| 72 | `neuralops/core/triton_kernels/gdn_scan.py` | 113 | `# Update S: S_t = beta_t * S + step * k_t^T @ err` | Ambiguous comment requiring manual confirmation | KEEP |
| 73 | `neuralops/core/triton_kernels/gdn_scan.py` | 273 | `# dS += q_t^T @ dy_t = outer product` | Ambiguous comment requiring manual confirmation | KEEP |
| 74 | `neuralops/core/triton_kernels/gdn_scan.py` | 295 | `# Path through inter_v = k_t @ S_prev:` | Ambiguous comment requiring manual confirmation | KEEP |
| 75 | `neuralops/core/triton_kernels/low_rank.py` | 105 | `# Pass 1: z = x @ V_mat` | Ambiguous comment requiring manual confirmation | KEEP |
| 76 | `neuralops/core/triton_kernels/low_rank.py` | 131 | `# Pass 2: y = z @ U_mat + bias` | Ambiguous comment requiring manual confirmation | KEEP |
| 77 | `neuralops/core/triton_kernels/sparse_attn.py` | 103 | `# Compute S_ij = Q_i @ K_j^T * sm_scale` | Ambiguous comment requiring manual confirmation | KEEP |
| 78 | `neuralops/core/triton_kernels/state_update.py` | 104 | `# Update state: h = h * dA + u * dB` | Ambiguous comment requiring manual confirmation | KEEP |
| 79 | `neuralops/data/synthetic_generator.py` | 66 | `# Sample token indices from power-law distribution` | Ambiguous comment requiring manual confirmation | KEEP |
| 80 | `neuralops/utils/replace_attention.py` | 426 | `# Pattern C: spectron/svd_compressed: (model_config, pretrained_attn=..., ranks=...)` | Ambiguous comment requiring manual confirmation | KEEP |
| 81 | `neuralops/utils/replace_attention.py` | 434 | `# Pattern E: soa_module: (model_config, k_rank=32, pretrained_attn=...)` | Ambiguous comment requiring manual confirmation | KEEP |
| 82 | `neuralops/__init__.py` | 39 | `# Attentions` | Ambiguous comment requiring manual confirmation | KEEP |
| 83 | `neuralops/__init__.py` | 90 | `# Compressors` | Ambiguous comment requiring manual confirmation | KEEP |
| 84 | `neuralops/__init__.py` | 124 | `# Reorganized Benchmarks & Sweepers` | Ambiguous comment requiring manual confirmation | KEEP |
| 85 | `neuralops/__init__.py` | 140 | `# Reorganized Evaluation & Verification` | Ambiguous comment requiring manual confirmation | KEEP |
| 86 | `neuralops/__init__.py` | 146 | `# Reorganized Trainer & Analysis` | Ambiguous comment requiring manual confirmation | KEEP |
| 87 | `neuralops/analysis/__init__.py` | 75 | `# Statistical testing suite` | Ambiguous comment requiring manual confirmation | KEEP |
| 88 | `neuralops/analysis/comparison/statistical_tests.py` | 38 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 89 | `neuralops/analysis/comparison/statistical_tests.py` | 40 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 90 | `neuralops/analysis/comparison/statistical_tests.py` | 81 | `# Reshape back` | Ambiguous comment requiring manual confirmation | KEEP |
| 91 | `neuralops/analysis/comparison/statistical_tests.py` | 86 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 92 | `neuralops/analysis/comparison/statistical_tests.py` | 88 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 93 | `neuralops/analysis/comparison/statistical_tests.py` | 105 | `# Average ranks` | Ambiguous comment requiring manual confirmation | KEEP |
| 94 | `neuralops/analysis/comparison/statistical_tests.py` | 133 | `# Test statistic` | Ambiguous comment requiring manual confirmation | KEEP |
| 95 | `neuralops/analysis/comparison/statistical_tests.py` | 194 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 96 | `neuralops/analysis/comparison/statistical_tests.py` | 196 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 97 | `neuralops/analysis/comparison/statistical_tests.py` | 223 | `# Normalised difference matrix: \` | UNCERTAIN | KEEP |
| 98 | `neuralops/analysis/comparison/statistical_tests.py` | 350 | `# Compute all pairwise z-statistics` | Ambiguous comment requiring manual confirmation | KEEP |
| 99 | `neuralops/analysis/comparison/statistical_tests.py` | 357 | `# Standard normal two-tailed p-values` | Ambiguous comment requiring manual confirmation | KEEP |
| 100 | `neuralops/analysis/comparison/statistical_tests.py` | 360 | `# Bergmann-Hommel dynamic correction` | Ambiguous comment requiring manual confirmation | KEEP |
| 101 | `neuralops/analysis/comparison/statistical_tests.py` | 363 | `# Sorted p-values` | Ambiguous comment requiring manual confirmation | KEEP |
| 102 | `neuralops/analysis/comparison/statistical_tests.py` | 370 | `# Adjusted p-values (step‑up)` | Ambiguous comment requiring manual confirmation | KEEP |
| 103 | `neuralops/analysis/comparison/statistical_tests.py` | 375 | `# Reconstruct matrix` | Ambiguous comment requiring manual confirmation | KEEP |
| 104 | `neuralops/analysis/comparison/statistical_tests.py` | 386 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 105 | `neuralops/analysis/comparison/statistical_tests.py` | 388 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 106 | `neuralops/analysis/comparison/statistical_tests.py` | 443 | `# Count discordant pairs` | Ambiguous comment requiring manual confirmation | KEEP |
| 107 | `neuralops/analysis/comparison/statistical_tests.py` | 450 | `# Continuity correction` | Ambiguous comment requiring manual confirmation | KEEP |
| 108 | `neuralops/analysis/comparison/statistical_tests.py` | 464 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 109 | `neuralops/analysis/comparison/statistical_tests.py` | 466 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 110 | `neuralops/analysis/comparison/statistical_tests.py` | 480 | `# Pooled standard deviation` | Ambiguous comment requiring manual confirmation | KEEP |
| 111 | `neuralops/analysis/comparison/statistical_tests.py` | 501 | `# Correction factor` | Ambiguous comment requiring manual confirmation | KEEP |
| 112 | `neuralops/analysis/comparison/statistical_tests.py` | 584 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 113 | `neuralops/analysis/comparison/statistical_tests.py` | 586 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 114 | `neuralops/analysis/comparison/statistical_tests.py` | 620 | `# Posterior sampling (non‑informative prior)` | Ambiguous comment requiring manual confirmation | KEEP |
| 115 | `neuralops/analysis/comparison/statistical_tests.py` | 672 | `# Counts` | Ambiguous comment requiring manual confirmation | KEEP |
| 116 | `neuralops/analysis/comparison/statistical_tests.py` | 676 | `# Posterior samples` | Ambiguous comment requiring manual confirmation | KEEP |
| 117 | `neuralops/analysis/comparison/statistical_tests.py` | 680 | `# Probabilities` | Ambiguous comment requiring manual confirmation | KEEP |
| 118 | `neuralops/analysis/comparison/statistical_tests.py` | 699 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 119 | `neuralops/analysis/comparison/statistical_tests.py` | 701 | `# ═════════════════════════════════════════════════════════════════════` | Ambiguous comment requiring manual confirmation | KEEP |
| 120 | `neuralops/analysis/comparison/statistical_tests.py` | 733 | `# Omnibus tests` | Ambiguous comment requiring manual confirmation | KEEP |
| 121 | `neuralops/analysis/generate_all_modules_analysis.py` | 1 | `#!/usr/bin/env python3` | Ambiguous comment requiring manual confirmation | KEEP |
| 122 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 1 | `#!/usr/bin/env python3` | Ambiguous comment requiring manual confirmation | KEEP |
| 123 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 72 | `# Compressor checks` | Ambiguous comment requiring manual confirmation | KEEP |
| 124 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 110 | `# Attention checks` | Ambiguous comment requiring manual confirmation | KEEP |
| 125 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 147 | `# Standardize column names` | Ambiguous comment requiring manual confirmation | KEEP |
| 126 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 150 | `# Ensure standard columns` | Ambiguous comment requiring manual confirmation | KEEP |
| 127 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 158 | `# Classifications` | Ambiguous comment requiring manual confirmation | KEEP |
| 128 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 199 | `# Baseline line` | Ambiguous comment requiring manual confirmation | KEEP |
| 129 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 208 | `# Add value labels` | Ambiguous comment requiring manual confirmation | KEEP |
| 130 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 262 | `# Separate by type` | Ambiguous comment requiring manual confirmation | KEEP |
| 131 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 290 | `# Annotate top performers` | Ambiguous comment requiring manual confirmation | KEEP |
| 132 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 467 | `# Table 1: Top Attention Mechanisms` | Ambiguous comment requiring manual confirmation | KEEP |
| 133 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 497 | `# Table 2: Top Compressors` | Ambiguous comment requiring manual confirmation | KEEP |
| 134 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 655 | `# Generate figures` | Ambiguous comment requiring manual confirmation | KEEP |
| 135 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 661 | `# Generate LaTeX tables` | Ambiguous comment requiring manual confirmation | KEEP |
| 136 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 665 | `# Generate Markdown report` | Ambiguous comment requiring manual confirmation | KEEP |
| 137 | `neuralops/analysis/layerwise_analyzer.py` | 180 | `# Match standard transformer layer patterns` | Ambiguous comment requiring manual confirmation | KEEP |
| 138 | `neuralops/analysis/layerwise_analyzer.py` | 225 | `# Explicitly release references` | Ambiguous comment requiring manual confirmation | KEEP |
| 139 | `neuralops/analysis/layerwise_analyzer.py` | 270 | `# Cleanup batch inputs` | Ambiguous comment requiring manual confirmation | KEEP |
| 140 | `neuralops/analysis/layerwise_analyzer.py` | 299 | `# Compute cosine similarity between adjacent layers` | Ambiguous comment requiring manual confirmation | KEEP |
| 141 | `neuralops/analysis/module_analysis/aggregators.py` | 26 | `# Strip common suffixes` | Ambiguous comment requiring manual confirmation | KEEP |
| 142 | `neuralops/analysis/plotting.py` | 14 | `# Default style` | Ambiguous comment requiring manual confirmation | KEEP |
| 143 | `neuralops/analysis/plotting.py` | 272 | `# Simple heatmap` | Ambiguous comment requiring manual confirmation | KEEP |
| 144 | `neuralops/attentions/adaptive/adaptive_span_attention.py` | 22 | `# Loss helpers` | Ambiguous comment requiring manual confirmation | KEEP |
| 145 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 50 | `# Ensure 4D broadcasting: (B, 1, L, rot_dim) or (1, 1, L, rot_dim)` | Ambiguous comment requiring manual confirmation | KEEP |
| 146 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 242 | `# Static layer index (not mutated inside forward)` | Ambiguous comment requiring manual confirmation | KEEP |
| 147 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 254 | `# LoRA adaptation configuration` | Ambiguous comment requiring manual confirmation | KEEP |
| 148 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 267 | `# Decoder & nested encoder detection` | Ambiguous comment requiring manual confirmation | KEEP |
| 149 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 487 | `# GPT-2 Output: c_proj` | Ambiguous comment requiring manual confirmation | KEEP |
| 150 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 624 | `# Weight verification: ensure weights are non-trivial` | Ambiguous comment requiring manual confirmation | KEEP |
| 151 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 728 | `# Previous tokens: includes block 0 (CLS at index 0)` | Ambiguous comment requiring manual confirmation | KEEP |
| 152 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 850 | `# Valid token mask` | Ambiguous comment requiring manual confirmation | KEEP |
| 153 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 855 | `# Static rectification decision (does not increment layer_idx across forward passes)` | Ambiguous comment requiring manual confirmation | KEEP |
| 154 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 863 | `# Dense attention pathway (full rectification)` | Ambiguous comment requiring manual confirmation | KEEP |
| 155 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 911 | `# Padding mask` | Ambiguous comment requiring manual confirmation | KEEP |
| 156 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 984 | `# Whole-model traversal` | Ambiguous comment requiring manual confirmation | KEEP |
| 157 | `neuralops/attentions/adaptive/eat.py` | 73 | `# LoRALinear (dtype-safe)` | Ambiguous comment requiring manual confirmation | KEEP |
| 158 | `neuralops/attentions/adaptive/eat.py` | 142 | `# EAT-FIX-08: dtype-safe` | Ambiguous comment requiring manual confirmation | KEEP |
| 159 | `neuralops/attentions/adaptive/eat.py` | 158 | `# EAT Block` | Ambiguous comment requiring manual confirmation | KEEP |
| 160 | `neuralops/attentions/adaptive/eat.py` | 311 | `# EAT-FIX-03: never mark DistilBERT-like as nested` | Ambiguous comment requiring manual confirmation | KEEP |
| 161 | `neuralops/attentions/adaptive/eat.py` | 395 | `# EAT-FIX-09` | Ambiguous comment requiring manual confirmation | KEEP |
| 162 | `neuralops/attentions/adaptive/eat.py` | 444 | `# copy_weights (EAT-FIX-11)` | Ambiguous comment requiring manual confirmation | KEEP |
| 163 | `neuralops/attentions/adaptive/eat.py` | 676 | `# EAT-FIX-15: warn (not raise)` | Ambiguous comment requiring manual confirmation | KEEP |
| 164 | `neuralops/attentions/adaptive/eat.py` | 697 | `# Forward` | Ambiguous comment requiring manual confirmation | KEEP |
| 165 | `neuralops/attentions/adaptive/eat.py` | 808 | `# Fast SDPA path` | Ambiguous comment requiring manual confirmation | KEEP |
| 166 | `neuralops/attentions/adaptive/eat.py` | 877 | `# EAT-FIX-01: no padding/truncation hack` | Ambiguous comment requiring manual confirmation | KEEP |
| 167 | `neuralops/attentions/adaptive/eat.py` | 895 | `# Injection` | Ambiguous comment requiring manual confirmation | KEEP |
| 168 | `neuralops/attentions/adaptive/eat.py` | 908 | `# EAT-FIX-04: recursion guard` | Ambiguous comment requiring manual confirmation | KEEP |
| 169 | `neuralops/attentions/adaptive/eat.py` | 975 | `# EAT-FIX-04: idempotency` | Ambiguous comment requiring manual confirmation | KEEP |
| 170 | `neuralops/attentions/adaptive/eat.py` | 1006 | `# Splice` | Ambiguous comment requiring manual confirmation | KEEP |
| 171 | `neuralops/attentions/adaptive/infini_attention.py` | 34 | `# RoPE` | Ambiguous comment requiring manual confirmation | KEEP |
| 172 | `neuralops/attentions/adaptive/infini_attention.py` | 86 | `# LoRALinear` | Ambiguous comment requiring manual confirmation | KEEP |
| 173 | `neuralops/attentions/adaptive/infini_attention.py` | 151 | `# SSM sub-branch` | Ambiguous comment requiring manual confirmation | KEEP |
| 174 | `neuralops/attentions/adaptive/infini_attention.py` | 204 | `# Decoder / format detection` | Ambiguous comment requiring manual confirmation | KEEP |
| 175 | `neuralops/attentions/adaptive/infini_attention.py` | 276 | `# Jamba-2 Block` | Ambiguous comment requiring manual confirmation | KEEP |
| 176 | `neuralops/attentions/adaptive/lola_attention.py` | 433 | `# GPT-2 fused c_attn` | Ambiguous comment requiring manual confirmation | KEEP |
| 177 | `neuralops/attentions/adaptive/lola_attention.py` | 489 | `# RoBERTa / BERT nested` | Ambiguous comment requiring manual confirmation | KEEP |
| 178 | `neuralops/attentions/adaptive/lola_attention.py` | 521 | `# Modern decoders + DistilBERT` | Ambiguous comment requiring manual confirmation | KEEP |
| 179 | `neuralops/attentions/adaptive/lola_attention.py` | 545 | `# Nested encoder detection` | Ambiguous comment requiring manual confirmation | KEEP |
| 180 | `neuralops/attentions/adaptive/lola_attention.py` | 659 | `# RoPE` | Ambiguous comment requiring manual confirmation | KEEP |
| 181 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 49 | `# RoPE` | Ambiguous comment requiring manual confirmation | KEEP |
| 182 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 96 | `# LoRALinear` | Ambiguous comment requiring manual confirmation | KEEP |
| 183 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 175 | `# Optional shared scorer (PTP-FIX-04)` | Ambiguous comment requiring manual confirmation | KEEP |
| 184 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 197 | `# PTP Attention` | Ambiguous comment requiring manual confirmation | KEEP |
| 185 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 233 | `# PTP-FIX-12: dimension resolution.` | Ambiguous comment requiring manual confirmation | KEEP |
| 186 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 290 | `# Projections` | Ambiguous comment requiring manual confirmation | KEEP |
| 187 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 336 | `# PTP-FIX-04: optional shared scorer.` | Ambiguous comment requiring manual confirmation | KEEP |
| 188 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 580 | `# Forward` | Ambiguous comment requiring manual confirmation | KEEP |
| 189 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 606 | `# PTP-FIX-03: single assignment.` | Ambiguous comment requiring manual confirmation | KEEP |
| 190 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 630 | `# Projections` | Ambiguous comment requiring manual confirmation | KEEP |
