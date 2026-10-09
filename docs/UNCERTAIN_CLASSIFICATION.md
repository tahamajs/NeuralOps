# Uncertain Classification Report (Part 3.3)

## Summary
- **Total items analyzed**: 190
- **Auto-classified as KEEP**: 131
- **Auto-classified as REMOVE**: 1
- **Truly uncertain**: 58

| # | File | Line | Comment | Classification | Rationale |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | `neuralops/core/_attention_extractors.py` | 70 | `# Also try nested attention.self.query` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 2 | `neuralops/core/attention_mask.py` | 48 | `# True = keep, False = mask` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 3 | `neuralops/core/rope.py` | 123 | `# In-place/fused expression: q_rot = q * cos + rotate_half(q) * sin` | `KEEP` | Mathematical formula / equation |
| 4 | `neuralops/core/selective_scan.py` | 169 | `# SC-FIX-04: exp(-inf) = 0 exactly; no nan_to_num needed.` | `KEEP` | Mathematical formula / equation |
| 5 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 522 | `# Output dense from outer container` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 6 | `neuralops/attentions/gated/gated_delta_net.py` | 834 | `# z_proj from q` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 7 | `neuralops/attentions/hybrid/fusion_attn.py` | 912 | `# take top-(k-1) from scores excluding block 0` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 8 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 177 | `# Shape alignment (unchanged from original)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 9 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 215 | `# Base SVD weights are already frozen via requires_grad=False on construction.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 10 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 641 | `# 1) Raw validity from padding (before any causal logic)` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 11 | `neuralops/attentions/hybrid/spectron_fp8_attention.py` | 844 | `# Init svd_U_q / svd_U_k from Q / K weights.` | `KEEP` | Mathematical formula / equation |
| 12 | `neuralops/attentions/linear/cosformer.py` | 406 | `# Zero contribution from padded keys.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 13 | `neuralops/attentions/linear/nystrom_attention.py` | 630 | `# Landmarks from all previous chunks (0..c_idx-1)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 14 | `neuralops/attentions/linear/rala_attention.py` | 644 | `# RALA-FIX-04: removed dead scale_factor=1.0` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 15 | `neuralops/attentions/linear/spectron.py` | 395 | `# Whitening factors per projection (from calibration)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 16 | `neuralops/attentions/linear/triton_cosformer.py` | 647 | `# BUG 5 FIX: fixed max_length from config` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 17 | `neuralops/attentions/modern/nha_attention.py` | 806 | `# NHA-FIX-02: do not silently return None` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 18 | `neuralops/attentions/sparse/cluster_attention.py` | 704 | `# Disincentivize padding tokens from pulling centroids` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 19 | `neuralops/attentions/sparse/lsh_attention.py` | 248 | `# LSH-10: set self.config explicitly` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 20 | `neuralops/attentions/sparse/reap_moe.py` | 1129 | `# RP-FIX-25: propagate n_experts, top_k from comp_config` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 21 | `neuralops/attentions/ssm/hymba_transx.py` | 452 | `# HX-FIX-07: diff_lambda init = -2.0 (sigmoid ≈ 0.12)` | `KEEP` | Mathematical formula / equation |
| 22 | `neuralops/attentions/ssm/jamba_info.py` | 389 | `# Stored as buffer (not parameter) because it's derived from lambda_init.` | `KEEP` | Mathematical formula / equation |
| 23 | `neuralops/attentions/ssm/mamba_block.py` | 932 | `# Recompute return format (MB-12)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 24 | `neuralops/attentions/ssm/mamba_svd_flash.py` | 738 | `# MSF-FIX-17: initialise SSM projections from W_q, W_k using SVD` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 25 | `neuralops/attentions/ssm/nexus.py` | 835 | `# copy_weights (unchanged from original)` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 26 | `neuralops/attentions/ssm/selective_ssm.py` | 96 | `# Delta, B, C projections from x_ssm` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 27 | `neuralops/attentions/ssm/selective_ssm.py` | 102 | `# A parameterization: A = -exp(A_log)` | `KEEP` | Mathematical formula / equation |
| 28 | `neuralops/attentions/ssm/upi_stuffed_mamba.py` | 683 | `# Float mask: -inf = invalid, 0 = valid` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 29 | `neuralops/attentions/svd/adaptive_svd_attention.py` | 705 | `# ASVD-FIX-06: always return (output, attn_weights)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 30 | `neuralops/attentions/svd/asvd_attention.py` | 253 | `# Factorized SVD projection: y = ((x @ V.T) * S) @ U.T + bias` | `KEEP` | Mathematical formula / equation |
| 31 | `neuralops/attentions/svd/svd_llm_v2_attention.py` | 214 | `# Device & Dtype detection from old_attn` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 32 | `neuralops/attentions/svd/svd_projection_attention.py` | 172 | `# Compute (x @ V^T) @ U^T = x @ (U @ V)^T` | `KEEP` | Mathematical formula / equation |
| 33 | `neuralops/benchmarks/discovery.py` | 91 | `# Family deduction from module name` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 34 | `neuralops/compression_engine/experimental/hashed_net_compressor.py` | 383 | `# lora_B @ lora_A / scaling = R (approx)` | `KEEP` | Mathematical formula / equation |
| 35 | `neuralops/compression_engine/experimental/liftr.py` | 104 | `# LIFTR-FIX-05: determine return signature once at init.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 36 | `neuralops/compression_engine/low_rank/come_compressor.py` | 119 | `# Tuple return` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 37 | `neuralops/compression_engine/low_rank/come_compressor.py` | 125 | `# List return` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 38 | `neuralops/compression_engine/low_rank/come_compressor.py` | 131 | `# Dict return` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 39 | `neuralops/compression_engine/low_rank/dora_qlora.py` | 120 | `# Magnitude parameter m = \` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 40 | `neuralops/compression_engine/low_rank/dora_qlora.py` | 126 | `# Effective directional weight: V = W + scaling * (B @ A)` | `KEEP` | Mathematical formula / equation |
| 41 | `neuralops/compression_engine/low_rank/grecodec_compressor.py` | 183 | `# Update U given V: min_U \` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 42 | `neuralops/compression_engine/low_rank/grecodec_compressor.py` | 192 | `# Update V given U: min_V \` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 43 | `neuralops/compression_engine/low_rank/nmf_compressor.py` | 125 | `# (H@V * W_sign) @ x = (H@V) @ (W_sign * x) element-wise? No —` | `KEEP` | Mathematical formula / equation |
| 44 | `neuralops/compression_engine/low_rank/oats_compressor.py` | 434 | `# Scale weights: W_scaled = W @ L_M` | `KEEP` | Mathematical formula / equation |
| 45 | `neuralops/compression_engine/low_rank/oats_compressor.py` | 484 | `# De-scale: W_comp = (L + S_mat) @ L_M^-1` | `KEEP` | Mathematical formula / equation |
| 46 | `neuralops/compression_engine/low_rank/salr.py` | 389 | `# Clear large tensors from GPU memory aggressively` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 47 | `neuralops/compression_engine/pruning/armor_pruner.py` | 395 | `# AR-FIX-05: Conv1D detection via class name.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 48 | `neuralops/compression_engine/pruning/armor_pruner.py` | 832 | `# AR-FIX-14: distinguish storage from effective.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 49 | `neuralops/compression_engine/pruning/flap.py` | 359 | `# FL-FIX-01: compute actual sparsity from mask.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 50 | `neuralops/compression_engine/pruning/proxsparse.py` | 504 | `# Final projection (mask from last iteration already applied).` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 51 | `neuralops/compression_engine/pruning/proxsparse.py` | 574 | `# PS-FIX-08: effective_parameter_ratio separate from compression_ratio.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 52 | `neuralops/compression_engine/pruning/wanda.py` | 44 | `# Compute Wanda importance metric: S = \` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 53 | `neuralops/compression_engine/quantization/awq.py` | 38 | `# Protect salient channels using activation magnitude: s = act^alpha` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 54 | `neuralops/compression_engine/quantization/awq.py` | 44 | `# Restore scaling: Q = Q' / diag(s)` | `KEEP` | Mathematical formula / equation |
| 55 | `neuralops/compression_engine/svd/aa_svd_compressor.py` | 301 | `# override from kwargs (highest priority after explicit args)` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 56 | `neuralops/compression_engine/svd/adaptive_ranks.py` | 750 | `# Update A: A = (W @ E @ B.T) @ inv(B @ E @ B.T + eps * I)` | `KEEP` | Mathematical formula / equation |
| 57 | `neuralops/compression_engine/svd/adaptive_ranks.py` | 762 | `# Update B: B = inv(A.T @ D @ A + eps * I) @ (A.T @ D @ W)` | `KEEP` | Mathematical formula / equation |
| 58 | `neuralops/compression_engine/svd/dynamic_rank.py` | 81 | `# Normalized singular probability mass function p_i = sigma_i / sum(sigma_j)` | `KEEP` | Mathematical formula / equation |
| 59 | `neuralops/compression_engine/svd/gfws_d_compressor.py` | 405 | `# Whiten weights: M = W * Cov(X)^{1/2}` | `KEEP` | Mathematical formula / equation |
| 60 | `neuralops/compression_engine/svd/gfws_d_compressor.py` | 412 | `# De-whiten: candidate = M_r * Cov(X)^{-1/2}` | `KEEP` | Mathematical formula / equation |
| 61 | `neuralops/compression_engine/svd/residual_svd_compressor.py` | 228 | `# Scale S_tail by adapter scaling factor gamma: (B @ A) * gamma = E_tail` | `KEEP` | Mathematical formula / equation |
| 62 | `neuralops/compression_engine/svd/swift_svd.py` | 410 | `# SV-FIX-03: raw_shape from base_layer, correctly oriented.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 63 | `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py` | 721 | `# KTL-FIX-13: total stored params = base + adapter.` | `KEEP` | Mathematical formula / equation |
| 64 | `neuralops/core/selective_scan.py` | 180 | `# Carry from previous chunk` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 65 | `neuralops/core/triton_kernels/fms_scan.py` | 298 | `# Gradients from y_t = num_t / denom_t` | `KEEP` | Mathematical formula / equation |
| 66 | `neuralops/core/triton_kernels/fms_scan.py` | 299 | `# dL/dnum_t = dy_t / denom_t` | `KEEP` | Mathematical formula / equation |
| 67 | `neuralops/core/triton_kernels/fms_scan.py` | 301 | `# dL/ddenom_t = - sum(dy_t * num_t) / (denom_t * denom_t)` | `KEEP` | Mathematical formula / equation |
| 68 | `neuralops/core/triton_kernels/fms_scan.py` | 311 | `# from num_t: sum_j S_t[i, j] * dnum_t[j]` | `KEEP` | Mathematical formula / equation |
| 69 | `neuralops/core/triton_kernels/fms_scan.py` | 313 | `# from denom_t: Z_t * ddenom_t` | `KEEP` | Mathematical formula / equation |
| 70 | `neuralops/core/triton_kernels/fms_scan.py` | 327 | `# from S_t: sum_j dS[i, j] * v_f[j]` | `KEEP` | Mathematical formula / equation |
| 71 | `neuralops/core/triton_kernels/fms_scan.py` | 328 | `# from Z_t: dZ[i]` | `REMOVE` | Commented-out code statement |
| 72 | `neuralops/core/triton_kernels/gdn_scan.py` | 113 | `# Update S: S_t = beta_t * S + step * k_t^T @ err` | `KEEP` | Mathematical formula / equation |
| 73 | `neuralops/core/triton_kernels/gdn_scan.py` | 273 | `# dS += q_t^T @ dy_t = outer product` | `KEEP` | Mathematical formula / equation |
| 74 | `neuralops/core/triton_kernels/gdn_scan.py` | 295 | `# Path through inter_v = k_t @ S_prev:` | `KEEP` | Mathematical formula / equation |
| 75 | `neuralops/core/triton_kernels/low_rank.py` | 105 | `# Pass 1: z = x @ V_mat` | `KEEP` | Mathematical formula / equation |
| 76 | `neuralops/core/triton_kernels/low_rank.py` | 131 | `# Pass 2: y = z @ U_mat + bias` | `KEEP` | Mathematical formula / equation |
| 77 | `neuralops/core/triton_kernels/sparse_attn.py` | 103 | `# Compute S_ij = Q_i @ K_j^T * sm_scale` | `KEEP` | Mathematical formula / equation |
| 78 | `neuralops/core/triton_kernels/state_update.py` | 104 | `# Update state: h = h * dA + u * dB` | `KEEP` | Mathematical formula / equation |
| 79 | `neuralops/data/synthetic_generator.py` | 66 | `# Sample token indices from power-law distribution` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 80 | `neuralops/utils/replace_attention.py` | 426 | `# Pattern C: spectron/svd_compressed: (model_config, pretrained_attn=..., ranks=...)` | `KEEP` | Mathematical formula / equation |
| 81 | `neuralops/utils/replace_attention.py` | 434 | `# Pattern E: soa_module: (model_config, k_rank=32, pretrained_attn=...)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 82 | `neuralops/__init__.py` | 39 | `# Attentions` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 83 | `neuralops/__init__.py` | 90 | `# Compressors` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 84 | `neuralops/__init__.py` | 124 | `# Reorganized Benchmarks & Sweepers` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 85 | `neuralops/__init__.py` | 140 | `# Reorganized Evaluation & Verification` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 86 | `neuralops/__init__.py` | 146 | `# Reorganized Trainer & Analysis` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 87 | `neuralops/analysis/__init__.py` | 75 | `# Statistical testing suite` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 88 | `neuralops/analysis/comparison/statistical_tests.py` | 38 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 89 | `neuralops/analysis/comparison/statistical_tests.py` | 40 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 90 | `neuralops/analysis/comparison/statistical_tests.py` | 81 | `# Reshape back` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 91 | `neuralops/analysis/comparison/statistical_tests.py` | 86 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 92 | `neuralops/analysis/comparison/statistical_tests.py` | 88 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 93 | `neuralops/analysis/comparison/statistical_tests.py` | 105 | `# Average ranks` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 94 | `neuralops/analysis/comparison/statistical_tests.py` | 133 | `# Test statistic` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 95 | `neuralops/analysis/comparison/statistical_tests.py` | 194 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 96 | `neuralops/analysis/comparison/statistical_tests.py` | 196 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 97 | `neuralops/analysis/comparison/statistical_tests.py` | 223 | `# Normalised difference matrix: \` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 98 | `neuralops/analysis/comparison/statistical_tests.py` | 350 | `# Compute all pairwise z-statistics` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 99 | `neuralops/analysis/comparison/statistical_tests.py` | 357 | `# Standard normal two-tailed p-values` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 100 | `neuralops/analysis/comparison/statistical_tests.py` | 360 | `# Bergmann-Hommel dynamic correction` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 101 | `neuralops/analysis/comparison/statistical_tests.py` | 363 | `# Sorted p-values` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 102 | `neuralops/analysis/comparison/statistical_tests.py` | 370 | `# Adjusted p-values (step‑up)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 103 | `neuralops/analysis/comparison/statistical_tests.py` | 375 | `# Reconstruct matrix` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 104 | `neuralops/analysis/comparison/statistical_tests.py` | 386 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 105 | `neuralops/analysis/comparison/statistical_tests.py` | 388 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 106 | `neuralops/analysis/comparison/statistical_tests.py` | 443 | `# Count discordant pairs` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 107 | `neuralops/analysis/comparison/statistical_tests.py` | 450 | `# Continuity correction` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 108 | `neuralops/analysis/comparison/statistical_tests.py` | 464 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 109 | `neuralops/analysis/comparison/statistical_tests.py` | 466 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 110 | `neuralops/analysis/comparison/statistical_tests.py` | 480 | `# Pooled standard deviation` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 111 | `neuralops/analysis/comparison/statistical_tests.py` | 501 | `# Correction factor` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 112 | `neuralops/analysis/comparison/statistical_tests.py` | 584 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 113 | `neuralops/analysis/comparison/statistical_tests.py` | 586 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 114 | `neuralops/analysis/comparison/statistical_tests.py` | 620 | `# Posterior sampling (non‑informative prior)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 115 | `neuralops/analysis/comparison/statistical_tests.py` | 672 | `# Counts` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 116 | `neuralops/analysis/comparison/statistical_tests.py` | 676 | `# Posterior samples` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 117 | `neuralops/analysis/comparison/statistical_tests.py` | 680 | `# Probabilities` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 118 | `neuralops/analysis/comparison/statistical_tests.py` | 699 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 119 | `neuralops/analysis/comparison/statistical_tests.py` | 701 | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 120 | `neuralops/analysis/comparison/statistical_tests.py` | 733 | `# Omnibus tests` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 121 | `neuralops/analysis/generate_all_modules_analysis.py` | 1 | `#!/usr/bin/env python3` | `KEEP` | Mathematical formula / equation |
| 122 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 1 | `#!/usr/bin/env python3` | `KEEP` | Mathematical formula / equation |
| 123 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 72 | `# Compressor checks` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 124 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 110 | `# Attention checks` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 125 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 147 | `# Standardize column names` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 126 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 150 | `# Ensure standard columns` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 127 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 158 | `# Classifications` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 128 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 199 | `# Baseline line` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 129 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 208 | `# Add value labels` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 130 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 262 | `# Separate by type` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 131 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 290 | `# Annotate top performers` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 132 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 467 | `# Table 1: Top Attention Mechanisms` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 133 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 497 | `# Table 2: Top Compressors` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 134 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 655 | `# Generate figures` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 135 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 661 | `# Generate LaTeX tables` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 136 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 665 | `# Generate Markdown report` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 137 | `neuralops/analysis/layerwise_analyzer.py` | 180 | `# Match standard transformer layer patterns` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 138 | `neuralops/analysis/layerwise_analyzer.py` | 225 | `# Explicitly release references` | `KEEP` | Paper citation or reference |
| 139 | `neuralops/analysis/layerwise_analyzer.py` | 270 | `# Cleanup batch inputs` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 140 | `neuralops/analysis/layerwise_analyzer.py` | 299 | `# Compute cosine similarity between adjacent layers` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 141 | `neuralops/analysis/module_analysis/aggregators.py` | 26 | `# Strip common suffixes` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 142 | `neuralops/analysis/plotting.py` | 14 | `# Default style` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 143 | `neuralops/analysis/plotting.py` | 272 | `# Simple heatmap` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 144 | `neuralops/attentions/adaptive/adaptive_span_attention.py` | 22 | `# Loss helpers` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 145 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 50 | `# Ensure 4D broadcasting: (B, 1, L, rot_dim) or (1, 1, L, rot_dim)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 146 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 242 | `# Static layer index (not mutated inside forward)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 147 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 254 | `# LoRA adaptation configuration` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 148 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 267 | `# Decoder & nested encoder detection` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 149 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 487 | `# GPT-2 Output: c_proj` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 150 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 624 | `# Weight verification: ensure weights are non-trivial` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 151 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 728 | `# Previous tokens: includes block 0 (CLS at index 0)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 152 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 850 | `# Valid token mask` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 153 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 855 | `# Static rectification decision (does not increment layer_idx across forward passes)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 154 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 863 | `# Dense attention pathway (full rectification)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 155 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 911 | `# Padding mask` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 156 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 984 | `# Whole-model traversal` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 157 | `neuralops/attentions/adaptive/eat.py` | 73 | `# LoRALinear (dtype-safe)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 158 | `neuralops/attentions/adaptive/eat.py` | 142 | `# EAT-FIX-08: dtype-safe` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 159 | `neuralops/attentions/adaptive/eat.py` | 158 | `# EAT Block` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 160 | `neuralops/attentions/adaptive/eat.py` | 311 | `# EAT-FIX-03: never mark DistilBERT-like as nested` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 161 | `neuralops/attentions/adaptive/eat.py` | 395 | `# EAT-FIX-09` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 162 | `neuralops/attentions/adaptive/eat.py` | 444 | `# copy_weights (EAT-FIX-11)` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 163 | `neuralops/attentions/adaptive/eat.py` | 676 | `# EAT-FIX-15: warn (not raise)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 164 | `neuralops/attentions/adaptive/eat.py` | 697 | `# Forward` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 165 | `neuralops/attentions/adaptive/eat.py` | 808 | `# Fast SDPA path` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 166 | `neuralops/attentions/adaptive/eat.py` | 877 | `# EAT-FIX-01: no padding/truncation hack` | `KEEP` | Mathematical formula / equation |
| 167 | `neuralops/attentions/adaptive/eat.py` | 895 | `# Injection` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 168 | `neuralops/attentions/adaptive/eat.py` | 908 | `# EAT-FIX-04: recursion guard` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 169 | `neuralops/attentions/adaptive/eat.py` | 975 | `# EAT-FIX-04: idempotency` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 170 | `neuralops/attentions/adaptive/eat.py` | 1006 | `# Splice` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 171 | `neuralops/attentions/adaptive/infini_attention.py` | 34 | `# RoPE` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 172 | `neuralops/attentions/adaptive/infini_attention.py` | 86 | `# LoRALinear` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 173 | `neuralops/attentions/adaptive/infini_attention.py` | 151 | `# SSM sub-branch` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 174 | `neuralops/attentions/adaptive/infini_attention.py` | 204 | `# Decoder / format detection` | `KEEP` | Mathematical formula / equation |
| 175 | `neuralops/attentions/adaptive/infini_attention.py` | 276 | `# Jamba-2 Block` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 176 | `neuralops/attentions/adaptive/lola_attention.py` | 433 | `# GPT-2 fused c_attn` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 177 | `neuralops/attentions/adaptive/lola_attention.py` | 489 | `# RoBERTa / BERT nested` | `KEEP` | Mathematical formula / equation |
| 178 | `neuralops/attentions/adaptive/lola_attention.py` | 521 | `# Modern decoders + DistilBERT` | `KEEP` | Mathematical formula / equation |
| 179 | `neuralops/attentions/adaptive/lola_attention.py` | 545 | `# Nested encoder detection` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 180 | `neuralops/attentions/adaptive/lola_attention.py` | 659 | `# RoPE` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 181 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 49 | `# RoPE` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 182 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 96 | `# LoRALinear` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 183 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 175 | `# Optional shared scorer (PTP-FIX-04)` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 184 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 197 | `# PTP Attention` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 185 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 233 | `# PTP-FIX-12: dimension resolution.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 186 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 290 | `# Projections` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
| 187 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 336 | `# PTP-FIX-04: optional shared scorer.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 188 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 580 | `# Forward` | `KEEP` | Short fragment (< 10 chars, kept for safety) |
| 189 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 606 | `# PTP-FIX-03: single assignment.` | `KEEP` | Explanatory comment (> 20 chars, capitalized) |
| 190 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 630 | `# Projections` | `UNCERTAIN` | Ambiguous comment requiring manual confirmation |
