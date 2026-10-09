# Commented Code Removal Plan (Phase 4 Dry Run)

## Summary

- **Total items analyzed**: 827
- **REMOVE candidates**: 37
- **KEEP items**: 600
- **UNCERTAIN items**: 190

> [!IMPORTANT]
> **SAFETY PROTOCOL**: This is a dry run report. **NO comments have been modified or deleted.**
> Waiting for explicit user approval before performing any modifications.

## Categorization Breakdown

### 1. REMOVE Candidates
Lines containing commented-out Python statements, unused debug prints, or abandoned code assignments.

| File:Line | Content | Category | Confidence | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `neuralops/core/pruning_utils.py:161` | `# H = (2/N) * X^T X` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:658` | `# past_len = N_kv - N_q is exact for both prefill and decode-with-cache.` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/attentions/linear/nystrom_attention.py:646` | `# F1 = softmax(Q_c @ K_landmarks^T)  → (B, H, C_len, m_prev)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/attentions/linear/nystrom_attention.py:650` | `# F2 = softmax(Q_landmarks @ K_landmarks^T)  → (B, H, m_prev, m_prev)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/attentions/linear/nystrom_attention.py:663` | `# F3 = softmax(Q_landmarks @ K_prev^T)  → (B, H, m_prev, start)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/attentions/ssm/goldfinch_ahn.py:263` | `# h_full = h_local + h_carry * prod_local` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/attentions/ssm/spectral_mamba.py:398` | `# output = gate * attn + (1 - gate) * mamba` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/low_rank/come_compressor.py:113` | `# residual = B @ (A @ t^T)^T + bias` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/low_rank/frandi_compressor.py:188` | `# B_new = B + delta @ A^+` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/low_rank/modegpt_compressor.py:422` | `# U_r = U[:, :r] · diag(S[:r]);  V_r = Vt[:r, :]` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/low_rank/svd_llm_compressor.py:204` | `# latent = x @ W_v.T + (optional LoRA_v)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/low_rank/svd_llm_compressor.py:214` | `# out = latent @ W_u.T + (optional LoRA_u)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/quantization/slicegpt.py:423` | `# W_reconstructed = W_pca[:, :keep] @ Q[:, :keep]^T` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/quantization/slicegpt_pca.py:651` | `# W_reconstructed = W_pca[:, :keep] @ V[:, :keep].T` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/svd/erc_svd.py:408` | `# fallback_to_original=False: proceed with high error (warned).` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/compression_engine/svd/swift_svd_act.py:470` | `# M = U_act^T W V_act` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/flash_attention.py:57` | `# dV = P^T * grad_out` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/flash_attention.py:60` | `# dP = grad_out * V^T` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/flash_attention.py:68` | `# dQ = dS * K` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/flash_attention.py:71` | `# dK = dS^T * Q` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/fused_backward.py:44` | `# P = softmax(scores)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/fused_backward.py:47` | `# dV = P^T @ dO` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/fused_backward.py:50` | `# dP = dO @ V^T` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/fused_backward.py:53` | `# dS = P * (dP - sum(dP * P))` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/fused_backward.py:58` | `# dQ = dS @ K` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/autograd/fused_backward.py:61` | `# dK = dS^T @ Q` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/fms_scan.py:124` | `# num = q_f @ S -> num[j] = sum_i q_f[i] * S[i, j]` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/fms_scan.py:126` | `# denom = q_f @ Z -> scalar` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/fms_scan.py:305` | `# num_t = q_f @ S_t  =>  dL/dS_t += q_f^T @ dnum_t` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/fms_scan.py:307` | `# denom_t = q_f @ Z_t => dL/dZ_t += q_f * ddenom_t` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/gdn_scan.py:103` | `# inter_v = k_t @ S -> (D,)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/gdn_scan.py:118` | `# y_t = q_t @ S -> (D,)` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/gdn_scan.py:313` | `# dS_prev = beta_t * dS - step * k_t @ k_t^T @ dS` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/gla_scan.py:102` | `# S_t = lam_t * S_{t-1} + k_t^T @ v_t` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/gla_scan.py:106` | `# Z_t = lam_t * Z_{t-1} + k_t` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/low_rank.py:188` | `# dz = grad_out @ U_mat.t() -> [M, Rank]` | `REMOVE` | 0.85 | Commented variable assignment |
| `neuralops/core/triton_kernels/low_rank.py:196` | `# z = x @ V_mat -> [M, Rank]` | `REMOVE` | 0.85 | Commented variable assignment |

### 2. UNCERTAIN Items
Lines that may be code fragments or brief notations requiring manual verification.

| File:Line | Content | Category | Confidence | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `neuralops/core/_attention_extractors.py:70` | `# Also try nested attention.self.query` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/attention_mask.py:48` | `# True = keep, False = mask` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/rope.py:123` | `# In-place/fused expression: q_rot = q * cos + rotate_half(q) * sin` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/selective_scan.py:169` | `# SC-FIX-04: exp(-inf) = 0 exactly; no nan_to_num needed.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:522` | `# Output dense from outer container` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/gated/gated_delta_net.py:834` | `# z_proj from q` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/hybrid/fusion_attn.py:912` | `# take top-(k-1) from scores excluding block 0` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:177` | `# Shape alignment (unchanged from original)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:215` | `# Base SVD weights are already frozen via requires_grad=False on construction.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:641` | `# 1) Raw validity from padding (before any causal logic)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/hybrid/spectron_fp8_attention.py:844` | `# Init svd_U_q / svd_U_k from Q / K weights.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/linear/cosformer.py:406` | `# Zero contribution from padded keys.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/linear/nystrom_attention.py:630` | `# Landmarks from all previous chunks (0..c_idx-1)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/linear/rala_attention.py:644` | `# RALA-FIX-04: removed dead scale_factor=1.0` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/linear/spectron.py:395` | `# Whitening factors per projection (from calibration)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/linear/triton_cosformer.py:647` | `# BUG 5 FIX: fixed max_length from config` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/modern/nha_attention.py:806` | `# NHA-FIX-02: do not silently return None` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/sparse/cluster_attention.py:704` | `# Disincentivize padding tokens from pulling centroids` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/sparse/lsh_attention.py:248` | `# LSH-10: set self.config explicitly` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/sparse/reap_moe.py:1129` | `# RP-FIX-25: propagate n_experts, top_k from comp_config` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/hymba_transx.py:452` | `# HX-FIX-07: diff_lambda init = -2.0 (sigmoid ≈ 0.12)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/jamba_info.py:389` | `# Stored as buffer (not parameter) because it's derived from lambda_init.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/mamba_block.py:932` | `# Recompute return format (MB-12)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/mamba_svd_flash.py:738` | `# MSF-FIX-17: initialise SSM projections from W_q, W_k using SVD` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/nexus.py:835` | `# copy_weights (unchanged from original)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/selective_ssm.py:96` | `# Delta, B, C projections from x_ssm` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/selective_ssm.py:102` | `# A parameterization: A = -exp(A_log)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:683` | `# Float mask: -inf = invalid, 0 = valid` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/svd/adaptive_svd_attention.py:705` | `# ASVD-FIX-06: always return (output, attn_weights)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/svd/asvd_attention.py:253` | `# Factorized SVD projection: y = ((x @ V.T) * S) @ U.T + bias` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/svd/svd_llm_v2_attention.py:214` | `# Device & Dtype detection from old_attn` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/svd/svd_projection_attention.py:172` | `# Compute (x @ V^T) @ U^T = x @ (U @ V)^T` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/benchmarks/discovery.py:91` | `# Family deduction from module name` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/experimental/hashed_net_compressor.py:383` | `# lora_B @ lora_A / scaling = R (approx)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/experimental/liftr.py:104` | `# LIFTR-FIX-05: determine return signature once at init.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/come_compressor.py:119` | `# Tuple return` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/come_compressor.py:125` | `# List return` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/come_compressor.py:131` | `# Dict return` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/dora_qlora.py:120` | `# Magnitude parameter m = \|\|W_0\|\|_c` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/dora_qlora.py:126` | `# Effective directional weight: V = W + scaling * (B @ A)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/grecodec_compressor.py:183` | `# Update U given V: min_U \|\|W - U V\|\|_F^2 => U = W V^T (V V^T)^{-1}` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/grecodec_compressor.py:192` | `# Update V given U: min_V \|\|W - U V\|\|_F^2 => V = (U^T U)^{-1} U^T W` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/nmf_compressor.py:125` | `# (H@V * W_sign) @ x = (H@V) @ (W_sign * x) element-wise? No —` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/oats_compressor.py:434` | `# Scale weights: W_scaled = W @ L_M` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/oats_compressor.py:484` | `# De-scale: W_comp = (L + S_mat) @ L_M^-1` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/low_rank/salr.py:389` | `# Clear large tensors from GPU memory aggressively` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/pruning/armor_pruner.py:395` | `# AR-FIX-05: Conv1D detection via class name.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/pruning/armor_pruner.py:832` | `# AR-FIX-14: distinguish storage from effective.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/pruning/flap.py:359` | `# FL-FIX-01: compute actual sparsity from mask.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/pruning/proxsparse.py:504` | `# Final projection (mask from last iteration already applied).` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/pruning/proxsparse.py:574` | `# PS-FIX-08: effective_parameter_ratio separate from compression_ratio.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/pruning/wanda.py:44` | `# Compute Wanda importance metric: S = \|W\| * \|\|X\|\|_2` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/quantization/awq.py:38` | `# Protect salient channels using activation magnitude: s = act^alpha` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/quantization/awq.py:44` | `# Restore scaling: Q = Q' / diag(s)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/aa_svd_compressor.py:301` | `# override from kwargs (highest priority after explicit args)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/adaptive_ranks.py:750` | `# Update A: A = (W @ E @ B.T) @ inv(B @ E @ B.T + eps * I)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/adaptive_ranks.py:762` | `# Update B: B = inv(A.T @ D @ A + eps * I) @ (A.T @ D @ W)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/dynamic_rank.py:81` | `# Normalized singular probability mass function p_i = sigma_i / sum(sigma_j)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/gfws_d_compressor.py:405` | `# Whiten weights: M = W * Cov(X)^{1/2}` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/gfws_d_compressor.py:412` | `# De-whiten: candidate = M_r * Cov(X)^{-1/2}` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/residual_svd_compressor.py:228` | `# Scale S_tail by adapter scaling factor gamma: (B @ A) * gamma = E_tail` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/svd/swift_svd.py:410` | `# SV-FIX-03: raw_shape from base_layer, correctly oriented.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py:721` | `# KTL-FIX-13: total stored params = base + adapter.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/selective_scan.py:180` | `# Carry from previous chunk` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:298` | `# Gradients from y_t = num_t / denom_t` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:299` | `# dL/dnum_t = dy_t / denom_t` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:301` | `# dL/ddenom_t = - sum(dy_t * num_t) / (denom_t * denom_t)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:311` | `# from num_t: sum_j S_t[i, j] * dnum_t[j]` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:313` | `# from denom_t: Z_t * ddenom_t` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:327` | `# from S_t: sum_j dS[i, j] * v_f[j]` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/fms_scan.py:328` | `# from Z_t: dZ[i]` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/gdn_scan.py:113` | `# Update S: S_t = beta_t * S + step * k_t^T @ err` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/gdn_scan.py:273` | `# dS += q_t^T @ dy_t = outer product` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/gdn_scan.py:295` | `# Path through inter_v = k_t @ S_prev:` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/low_rank.py:105` | `# Pass 1: z = x @ V_mat` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/low_rank.py:131` | `# Pass 2: y = z @ U_mat + bias` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/sparse_attn.py:103` | `# Compute S_ij = Q_i @ K_j^T * sm_scale` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/core/triton_kernels/state_update.py:104` | `# Update state: h = h * dA + u * dB` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/data/synthetic_generator.py:66` | `# Sample token indices from power-law distribution` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/utils/replace_attention.py:426` | `# Pattern C: spectron/svd_compressed: (model_config, pretrained_attn=..., ranks=...)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/utils/replace_attention.py:434` | `# Pattern E: soa_module: (model_config, k_rank=32, pretrained_attn=...)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/__init__.py:39` | `# Attentions` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/__init__.py:90` | `# Compressors` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/__init__.py:124` | `# Reorganized Benchmarks & Sweepers` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/__init__.py:140` | `# Reorganized Evaluation & Verification` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/__init__.py:146` | `# Reorganized Trainer & Analysis` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/__init__.py:75` | `# Statistical testing suite` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:38` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:40` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:81` | `# Reshape back` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:86` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:88` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:105` | `# Average ranks` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:133` | `# Test statistic` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:194` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:196` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:223` | `# Normalised difference matrix: \|rank_i - rank_j\| / CD` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:350` | `# Compute all pairwise z-statistics` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:357` | `# Standard normal two-tailed p-values` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:360` | `# Bergmann-Hommel dynamic correction` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:363` | `# Sorted p-values` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:370` | `# Adjusted p-values (step‑up)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:375` | `# Reconstruct matrix` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:386` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:388` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:443` | `# Count discordant pairs` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:450` | `# Continuity correction` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:464` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:466` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:480` | `# Pooled standard deviation` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:501` | `# Correction factor` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:584` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:586` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:620` | `# Posterior sampling (non‑informative prior)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:672` | `# Counts` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:676` | `# Posterior samples` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:680` | `# Probabilities` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:699` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:701` | `# ═════════════════════════════════════════════════════════════════════` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/comparison/statistical_tests.py:733` | `# Omnibus tests` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_all_modules_analysis.py:1` | `#!/usr/bin/env python3` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:1` | `#!/usr/bin/env python3` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:72` | `# Compressor checks` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:110` | `# Attention checks` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:147` | `# Standardize column names` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:150` | `# Ensure standard columns` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:158` | `# Classifications` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:199` | `# Baseline line` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:208` | `# Add value labels` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:262` | `# Separate by type` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:290` | `# Annotate top performers` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:467` | `# Table 1: Top Attention Mechanisms` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:497` | `# Table 2: Top Compressors` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:655` | `# Generate figures` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:661` | `# Generate LaTeX tables` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:665` | `# Generate Markdown report` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/layerwise_analyzer.py:180` | `# Match standard transformer layer patterns` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/layerwise_analyzer.py:225` | `# Explicitly release references` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/layerwise_analyzer.py:270` | `# Cleanup batch inputs` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/layerwise_analyzer.py:299` | `# Compute cosine similarity between adjacent layers` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/module_analysis/aggregators.py:26` | `# Strip common suffixes` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/plotting.py:14` | `# Default style` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/analysis/plotting.py:272` | `# Simple heatmap` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/adaptive_span_attention.py:22` | `# Loss helpers` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:50` | `# Ensure 4D broadcasting: (B, 1, L, rot_dim) or (1, 1, L, rot_dim)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:242` | `# Static layer index (not mutated inside forward)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:254` | `# LoRA adaptation configuration` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:267` | `# Decoder & nested encoder detection` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:487` | `# GPT-2 Output: c_proj` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:624` | `# Weight verification: ensure weights are non-trivial` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:728` | `# Previous tokens: includes block 0 (CLS at index 0)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:850` | `# Valid token mask` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:855` | `# Static rectification decision (does not increment layer_idx across forward passes)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:863` | `# Dense attention pathway (full rectification)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:911` | `# Padding mask` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:984` | `# Whole-model traversal` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:73` | `# LoRALinear (dtype-safe)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:142` | `# EAT-FIX-08: dtype-safe` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:158` | `# EAT Block` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:311` | `# EAT-FIX-03: never mark DistilBERT-like as nested` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:395` | `# EAT-FIX-09` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:444` | `# copy_weights (EAT-FIX-11)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:676` | `# EAT-FIX-15: warn (not raise)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:697` | `# Forward` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:808` | `# Fast SDPA path` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:877` | `# EAT-FIX-01: no padding/truncation hack` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:895` | `# Injection` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:908` | `# EAT-FIX-04: recursion guard` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:975` | `# EAT-FIX-04: idempotency` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/eat.py:1006` | `# Splice` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/infini_attention.py:34` | `# RoPE` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/infini_attention.py:86` | `# LoRALinear` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/infini_attention.py:151` | `# SSM sub-branch` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/infini_attention.py:204` | `# Decoder / format detection` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/infini_attention.py:276` | `# Jamba-2 Block` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/lola_attention.py:433` | `# GPT-2 fused c_attn` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/lola_attention.py:489` | `# RoBERTa / BERT nested` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/lola_attention.py:521` | `# Modern decoders + DistilBERT` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/lola_attention.py:545` | `# Nested encoder detection` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/lola_attention.py:659` | `# RoPE` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:49` | `# RoPE` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:96` | `# LoRALinear` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:175` | `# Optional shared scorer (PTP-FIX-04)` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:197` | `# PTP Attention` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:233` | `# PTP-FIX-12: dimension resolution.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:290` | `# Projections` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:336` | `# PTP-FIX-04: optional shared scorer.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:580` | `# Forward` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:606` | `# PTP-FIX-03: single assignment.` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:630` | `# Projections` | `UNCERTAIN` | 0.50 | Ambiguous comment requiring manual confirmation |

### 3. KEEP Items (Preserved Scientific & Architectural Context)
Preserved items include mathematical equations, paper derivations, algorithmic invariants, and docstring-like explanations.

| File:Line | Content | Category | Confidence | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `neuralops/__init__.py:18` | `# Transparent Import Hook for Cleaned Submodule Aliases` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/__init__.py:168` | `# Symbol to (module_path, attr_name) mapping for fast, lazy resolution` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:32` | `# Styling constants for publication-grade figures (matching NeurIPS / ICML / ICLR aesthetics)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:165` | `# Baseline comparison (if full attention is present)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:229` | `# Custom legend for families present` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/layerwise_analyzer.py:35` | `# Online moments for kurtosis and variance` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:39` | `# Covariance sketch for effective rank (subsampled rank estimation)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:97` | `# 4. Running Subsampled Covariance for Effective Rank` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:199` | `# Fallback: if no digit-indexed layers found, look for any layer module` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/__init__.py:90` | `# 1. Backward compatibility shim for sys.modules and import hooks (old flat import paths -> new subpackages)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/__init__.py:129` | `# Pre-populate sys.modules aliases (keep 'replace' dynamic on import so it doesn't affect canonical registry keys)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/__init__.py:22` | `# 1. Backward compatibility shim for sys.modules and import hooks (old flat import paths -> new subpackages)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/_attention_injectors.py:94` | `# Ensure old_attn has canonical projection attributes for custom modules inspecting pretrained_attn` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/_attention_surgery.py:51` | `# Regex matcher for unified multi-head projection layers (Falcon, StarCoder, GPT-2, GPT-NeoX, LLaMA-fused)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/_attention_surgery.py:171` | `# Public functional alias for fused QKV slicing` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/_attention_surgery.py:364` | `# Public functional alias for setting low-rank projection weights` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/_attention_wrapper.py:397` | `# Add repeat_kv for GQA:` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/adapters.py:129` | `# 4. Fallback for Fused GPT-2 c_attn (must precede positional fallback)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/adapters.py:145` | `# 4b. Fallback for Qwen3.5 / fused in_proj_qkv` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/adapters.py:201` | `#    Only valid if at least 3 children exist (the typical pattern for BERT/RoBERTa variants).` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/adapters.py:212` | `# 6. Fallback for 'o' projection when replacing leaf self-attention without local out_proj` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/architecture.py:94` | `# Heuristic 1: Module class name indicates attention or linear-recurrent block` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/attention_mask.py:52` | `# Check if it's already negative additive or 0/1 binary` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/attention_mask.py:57` | `# 1.0 = keep, 0.0 = mask` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/attention_mask.py:84` | `# HuggingFace extended attention mask already contains 0.0 for unmasked and -10000.0 for masked` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_family.py:191` | `# position_embedding: rotary for modern decoders, absolute for BERT/GPT-2,` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/model_utils.py:34` | `# Weight‑copying for attention modules` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_utils.py:115` | `# Alias for backward-compatibility with core lazy imports` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/model_wrapper.py:29` | `# 1. Custom config class to hold our additional parameters` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/model_wrapper.py:63` | `# 2. The unified model class with HubMixin` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/model_wrapper.py:166` | `# 3. Re‑apply the attention replacement if a type was stored` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/model_wrapper.py:224` | `# 3. Auto‑class registration – so that AutoModel works automatically` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/model_wrapper.py:228` | `# if the config.json contains our custom keys. (Simplified: always return UnifiedTransformerModel)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_wrapper.py:229` | `# In practice, you'd only do this if the config indicates it's a unified model.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_wrapper.py:230` | `# For simplicity, we override the auto class behavior for "unified_transformer".` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_wrapper.py:231` | `# However, to avoid conflicts, we'll register a separate auto class name.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_wrapper.py:234` | `# This allows AutoModel.from_pretrained("your-repo/your-model") to return a UnifiedTransformerModel` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_wrapper.py:235` | `# if the config.json contains the appropriate auto_map. We'll set that up when saving.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/pruning_utils.py:105` | `# Try input_ids (for language models) or continuous hidden states (for sequential/vision)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/pruning_utils.py:187` | `# 2. Standard LU inversion with progressive damping boost if needed` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/pruning_utils.py:252` | `# Handle remainder if d_in is not divisible by 4 (H5: strict 50% target)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/pruning_utils.py:278` | `# Handle remainder if d_out is not divisible by 4 (H5: strict 50% target)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/pruning_utils.py:455` | `# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/selective_scan.py:130` | `# dA: (B, D, L, N), always <= 0` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/selective_scan.py:143` | `# Enforce memory bound even if user gave explicit chunk_size.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/selective_scan.py:172` | `# h_intra[b, d, j, n] = sum_i decay[b, d, j, i, n] * dBu_c[b, d, i, n]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/t5_compat.py:188` | `# 3. 2D mask handling if provided` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/data/base.py:62` | `# Fallback auto-detection for text columns if specified column is missing` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/data/prepare_datasets.py:16` | `# Prepare all benchmark datasets for GPU server` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/data/synthetic_generator.py:56` | `# FIX: Compute exact Zipfian rank probabilities: P(k) = (1 / k^s) / zeta(s)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/data/synthetic_generator.py:181` | `# RESEARCH FIX: Compatibility alias for language modeling datasets` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/eval_data.py:24` | `# Caches for tokenized evaluation batches and loaded huggingface datasets` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/eval_patcher.py:37` | `# Cache storing (state_dict_cpu, config) for clean base model instantiation` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/eval_reporter.py:154` | `# Fallback if filelock is absent` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/full_mnli_validation.py:69` | `# Use 'premise'+'hypothesis' for MNLI, 'sentence' for SST-2` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/evaluation/full_mnli_validation.py:178` | `# Compute threshold for full split` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/full_mnli_validation.py:190` | `# Read sweep CSV — select PASS modules with acc >= min_acc_filter` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/evaluation/full_mnli_validation.py:215` | `# Load base model fresh for each module` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/evaluation/full_mnli_validation.py:241` | `# Load saved module checkpoint if it exists` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/label_quality.py:76` | `# 3. Per‑class breakdown` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/label_quality.py:115` | `# Average threshold for each class: t_k = mean prob of class k when k is predicted` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/evaluation/label_quality.py:124` | `# Confident joint: count how many times (given_label = i, true_label = j)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/label_quality.py:145` | `# For each sample, the label quality score = 1 - P(mislabeled)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/evaluation/label_quality.py:149` | `# A sample is a label issue if self‑confidence is below the class threshold` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/label_quality.py:178` | `# Subsample for performance` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/retraining.py:105` | `# Reset daily counter if it's a new day` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/run_lm_eval_harness.py:152` | `# Fallback: record all scalar numbers if standard patterns were not matched` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/experiment_logging/integration.py:47` | `# Save config snapshot for reproducibility` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/experiment_logging/logger.py:21` | `# Active listeners registry for clean shutdown` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/experiment_logging/logger.py:147` | `# Stream logs to stdout (typically rank 0 only or all ranks if formatted)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/experiment_logging/logger.py:170` | `# FIX: Implement Python logging with QueueHandler and QueueListener for thread safety and non-blocking I/O` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/experiment_utils/tasks.py:133` | `# Called by the orchestrator for log messages` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/integrations/hf_automodel.py:18` | `# Extract hidden dimension if available` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/main.py:74` | `# Point to local cache if present — checked in priority order, first match wins` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/main.py:210` | `# Crop dataset to max_train_samples (e.g. 1000) if requested for fast benchmarking` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/main.py:296` | `# Full end-to-end parameter fine-tuning enabled for all attention mechanisms` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/results_saver/config.py:75` | `# ── Cloud credentials / endpoints (for S3, GCP, etc.) ─────` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/runners/run_continuous_orchestrator.py:86` | `# Check if run_batch3_gpu_sweep.py is currently running` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/runners/run_sweep.py:48` | `# Prepend unknown_args so sys.argv is manipulated correctly for run_orchestrator` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/tracking/backends.py:27` | `# Convenience constants for configuration` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/tracking/backends.py:95` | `# Append to file immediately for resilience` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/utils/hf_git_exporter.py:189` | `# Check if table already exists with headers` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/utils/hf_git_exporter.py:293` | `# 3. Prune large uncompressed binary if present to keep upload lean (~90 MB)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/utils/layer_norm_fix.py:130` | `# FIX: Explicitly cast accumulators to torch.float32 for mean and variance to prevent FP16 variance underflow` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/utils/layer_norm_fix.py:148` | `# Aliases for explicit usage` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/utils/lora.py:49` | `# Trainable LoRA parameters are kept in FP32 for numerical stability` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/utils/lora.py:215` | `# Check if this child's name matches any target module` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/utils/replace_attention.py:244` | `# 4. Extract projection layers from self._wrapped_attention` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/utils/replace_attention.py:378` | `# Build a default ranks dict for spectron/svd_compressed which require it` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/utils/replace_attention.py:537` | `# Copy weights if supported` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:128` | `# Sum of aligned ranks per algorithm` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:484` | `# Interpretation: small=0.2, medium=0.5, large=0.8` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:557` | `# Normalise so that >= 0.5 (A is better if A12 > 0.5)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:669` | `# Using the method from Benavoli et al.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/module_analysis/parsers.py:27` | `# Map: lower_name -> dict(dataset -> list of rows)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/__init__.py:146` | `# 2. Import all subpackages and re-export classes at top level` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/adaptive_span_attention.py:357` | `# FIXED: Failure Mode 1 (Mode Collapse): Enforce min_span = 32 floor` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/adaptive_span_attention.py:358` | `# to prevent the attention span from collapsing to S -> 1 token ("eseseses..." loops)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:953` | `# Merge heads: (B, H, L, head_dim) -> (B, L, all_head_size)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/infini_attention.py:624` | `# FIX 1: Return the clean parallel delta ONLY (Outer block will add the residual!)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/squeezing_heads_distill.py:242` | `# SQ2-FIX-01: derive head_keep_ratio from `rank` if not explicitly given.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/squeezing_heads_distill.py:246` | `# If rank <= num_heads, interpret as "keep this many heads"` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/squeezing_heads_distill.py:605` | `# O must map from kept_q_head_size back to hidden.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/squeezing_heads_distill.py:660` | `# Case B: source has more heads (MHA → GQA pooling)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/token_compressed_attention.py:629` | `# Average over heads and query positions -> importance per key` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/token_compressed_attention.py:714` | `# Soft mask from softmax` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/token_compressed_attention.py:717` | `# Straight-through: forward = hard, backward = soft` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/token_compressed_attention.py:877` | `# ste_mask: (B, L) -> reduce to (B, 1, 1, L) for broadcasting` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/token_compressed_attention.py:881` | `# Blend: attn = attn + (ste_norm - ste_norm.detach()) * small_factor` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/token_compressed_attention.py:909` | `# Return cache if requested` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/fast_kernels/adasplash_attention.py:241` | `# FIXED: Failure Mode 1 (Mode Collapse): Default alpha = 1.0 (exact softmax)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/fast_kernels/adasplash_attention.py:242` | `# Prevents sparse entmax from zeroing out the entire distribution tail, stopping degenerate "eseseses..." loops.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/fast_kernels/flash_attention3.py:711` | `# Frobenius norm verification (>= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:762` | `# Frobenius norm verification (>= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:870` | `# out_tensor is (B, L, H, Dh) -> transpose to (B, H, L, Dh) for head masking` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/fast_kernels/soa_attn.py:278` | `# Probe head_dim from source attention if available` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/fast_kernels/xattention.py:558` | `# Replacing self.out_proj with nn.Identity avoids 7M redundant weights and parameter expansion.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/fast_kernels/xattention.py:563` | `# Frobenius norm verification (>= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/fast_kernels/xattention.py:779` | `# محاسبه Attention با O(N * k) حافظه و پردازش` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/decay_linear_attention.py:673` | `# Frobenius norm verification (>= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/decay_linear_attention.py:718` | `# mask is [B, L] boolean tensor where True = valid token` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/gated/decay_linear_attention.py:887` | `# Data-dependent decay lambdas in (0, 1) with bounded negative log decay bias` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/decay_linear_attention.py:898` | `# 5. True O(L) Linear Decay Scan` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/gated_delta_net.py:173` | `# ۲. نرمال‌سازی با eps=1e-8 (محافظت قطعی در برابر خطای NaN)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/gated_delta_net.py:177` | `# ۳. ضرب ماتریسی همزمان برای تمام لایه‌ها و بچ‌ها -> خروجی: (Num_Layers, Batch, Seq_Len, Seq_Len)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/gated_linear_attention.py:318` | `# FIX 6: decay_gate bias init = 2.0` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/gated_linear_attention.py:325` | `# FIX 1: log_decay init = 2.0` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/gated_linear_attention.py:329` | `# NEW in v3: learnable state_bias (init=0)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/gated/gated_linear_attention.py:335` | `# FIX 3: state_norm init = identity` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/gated_linear_attention.py:341` | `# FIX 2: output_scale init = 0.1` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/gated_linear_attention.py:543` | `# FIX 5: z_proj = 0.5 * W_q` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/hgrn2_gated_deltanet.py:162` | `# Initialized to ~98% HGRN, 2% DeltaNet via bias = 4.0` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/gated/hgrn2_gated_deltanet.py:299` | `# Uses branch devices to return zero safely` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/hgrn2_gated_layer.py:359` | `# FIX 11: map W_q → g_proj and W_k → input_gate` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/hgrn2_gated_outer.py:221` | `# FIX 15: strong signal from old_attn` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/hla_attention.py:426` | `# Attention weights with exponential decay: scores = (q @ k^T) * decay_mat` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/gated/hla_attention.py:430` | `# Numerator: y_num = scores @ v` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/gated/hla_attention.py:433` | `# Denominator: y_denom = sum_j scores` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/hybrid/fusion_attn.py:893` | `# A more direct approach: for each query block i, only keep scores over j <= i` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/fusion_attn.py:931` | `# gather: (B, H, nb, top_k, bs, D) → reshape to (B, H, nb, top_k*bs, D)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/fusion_attn.py:941` | `# reshape k_blocks to (B,H,nb,bs*D) and gather along dim=2 with flattened indices` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/hybrid/fusion_attn.py:1012` | `# FA-FIX-10: return block attention weights when requested` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:150` | `# FIX 2: rename A -> lora_B, B -> lora_A with PEFT shapes.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:257` | `# LoRA branch: x @ lora_A^T @ lora_B^T` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:663` | `# Causal: query q can attend to key k iff abs_k <= abs_q` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:703` | `#    FlashAttention / Triton kernel to actually get O(N*W) memory).` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:713` | `# 3) Local band: \|abs_q - abs_k\| <= W` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:717` | `# 4) Causal band: query at abs_q can attend to key at abs_k iff abs_k <= abs_q.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:720` | `#    because abs_q = N_k-1 >= every abs_k.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:731` | `# 6) Safe softmax: rows with no valid key → zero output (not uniform).` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:991` | `# Always return a plain 2-tuple so `out, weights = attn(...)` unpacking works` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/hybrid/halo_regla.py:277` | `# Output: (B, L, num_heads) → sigmoid → per-head gating` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py:890` | `# (B, 1, N_q, N_kv) -> OR over query dim -> (B, N_kv)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/landmark_local_attention.py:520` | `# Block-wise Sliding Window Attention (True O(N) memory)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/landmark_local_attention.py:564` | `# valid_mask shape: (B, N_pad) → broadcast against 2W dim` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/landmark_local_attention.py:930` | `# 2. Local branch (block-wise sliding window, true O(N) memory)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/optimized_hybrid_attention.py:706` | `# comp_chunk_mask[n, r] = True iff chunk_starts[r] <= n <= chunk_ends[r]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/optimized_hybrid_attention.py:734` | `# chunk_ends[r] <= i` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/refined_landmark_attention.py:391` | `# Gate init = +2.0 → sigmoid(2) ≈ 0.88 favors LANDMARK branch` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/refined_landmark_attention.py:559` | `# Block-wise Sliding Window Attention (True O(N) memory)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/refined_landmark_attention.py:603` | `# valid_mask shape: (B, N_pad) → broadcast against 2W dim` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/refined_landmark_attention.py:967` | `# 2. Local branch (block-wise sliding window, true O(N) memory)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/refined_landmark_attention.py:970` | `# 3. Gate fusion: init sigmoid(+2.0)≈0.88 → favors LANDMARK branch` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/regla_attention.py:908` | `# RE-19: propagate kernel_size, chunk_size, rank from comp_config` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/spectron_fp8_attention.py:932` | `# Flatten heads → (B, N, h*d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/spectron_fp8_attention.py:937` | `# Low-rank projection → (B, N, rank)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/spectron_fp8_attention.py:967` | `# (B, N, h*d) → (B, h, N, d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py:131` | `# V5-FIX-09: local_window = full width of the sliding window.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py:132` | `# Effective radius = local_window // 2 (queried keys are \|q - k\| <= radius).` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py:142` | `# FIXED: Set rff_trainable = True so the global projection buffers become trainable nn.Parameters` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py:146` | `# Chunk size for causal kernel linear attention (lower = less memory, slower).` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py:177` | `# FIXED: Capacity Bottleneck: Increased default lora_rank from 16 to 64` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py:929` | `# (B, 1, N_q, N_kv) -> OR over query dim -> (B, N_kv)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:287` | `# Optional orthogonal init for variance reduction when m <= d` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:325` | `# FIX 5: if window covers full sequence, use radius = N_kv (full context)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:334` | `# FIX 6: avoid .item() — compute q_min/q_max from past_len + indices` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:979` | `# FIX 2: use causal version when is_decoder=True` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/linear/chela_slay.py:309` | `# 2. Inter-chunk attention from running state` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/linear/chela_slay.py:699` | `# Adaptive gate — starts high at gate_init_bias (8.0 -> ~99.97% dense)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/linformer_attention.py:859` | `# E, F: (k_proj_dim, max_seq_len) → (k_eff, L)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/linformer_attention.py:906` | `# For each query position i, allow projected keys whose support range includes <= i` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/linformer_attention.py:907` | `# Simplified: allow key j if j <= i (proportional to support)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/nystrom_attention.py:721` | `# Causal (decoder) → Chunked Causal Nyström` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/performer_attention.py:442` | `# If explicit value passed, use it. Otherwise infer from source_attn.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/linear/primal_attention.py:201` | `# PR-FIX-03: guard self.rank against None` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/primal_attention.py:678` | `# PR-FIX-05: return softmax probs, not logits.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/rpc_attention.py:789` | `# (B, H, L, head_dim) @ (H, head_dim, r) -> (B, H, L, r)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/spectraformer_attention.py:520` | `# Preserve existing LoRA adapters from source projections` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/linear/spectraformer_attention.py:649` | `# Bidirectional linear attention: exact O(N·r·d) computation` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tlinformer_attention.py:693` | `# (H,) → (1, H, 1, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tlinformer_attention.py:696` | `# (B, H) → (B, H, 1, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tlinformer_attention.py:699` | `# Interpret as (B, H, 1) → (B, H, 1, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tlinformer_attention.py:818` | `# Degenerate case: N ≤ rank → dense attention (identity-ish projection)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tlinformer_attention.py:868` | `# context: (B, H, N, r) @ (B, H, r, Dh) → (B, H, N, Dh)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tlinformer_attention.py:875` | `# FIX 4 (output_attentions): return None or approximate reconstruction` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/linear/tlinformer_attention.py:879` | `# Reconstruction: attn @ P_k  where P_k: (r, N) → (B, H, N, N)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/tpa_attention.py:563` | `# Extract weights from various architectures` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/linear/triton_cosformer.py:180` | `# 4. Inter-chunk projection from running state:` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/linear/triton_cosformer.py:400` | `# Strictly O(N * d^2) PyTorch Fallback for CPU / Mac MPS (Zero N x N memory footprint)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/triton_cosformer.py:414` | `# Chunked causal scan: memory O(N * d), strictly eliminating O(N^2) allocations` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/triton_cosformer.py:788` | `# RoPE from kwargs if not explicit` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/linear/triton_cosformer.py:825` | `# Shape (B, N) -> (B, 1, N, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/triton_cosformer.py:843` | `# of routing through triton_dispatch → linear_attn.fwd, because:` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/linear/triton_cosformer.py:845` | `#   - That path was broken (ctx.save_for_backward not called before return)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/modern/gqa_attention.py:259` | `# GQA-FIX-16: derive num_kv_heads from `rank` when not explicitly given` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/gqa_attention.py:264` | `#   3. `rank` argument (1 <= rank <= num_heads)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/gqa_attention.py:266` | `#   5. default = num_heads // 4` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/modern/gqa_attention.py:768` | `# FIXED: Removed destructive torch.nan_to_num(..., nan=-1e4) before softmax` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/gqa_attention.py:820` | `# Injection (GQA-FIX-16: propagate rank → num_kv_heads)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/mla_attention.py:611` | `# naturally has the correct norm: \|\|U_r Σ_r V_r^T\|\|_F = \|\|Σ_r\|\|_F.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/modern/qcqa_attention.py:341` | `# QC-FIX-20: router maps q's last dim → num_kv_heads` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/qcqa_attention.py:693` | `# QC-FIX-13: Dynamic routing → head-level KV synthesis → MHA` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/qcqa_attention.py:696` | `# q: (B, L, H_q, d) → router → (B, L, H_q, H_kv)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/qcqa_attention.py:701` | `# This keeps the routing input-dependent (from q) while remaining` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/modern/qcqa_attention.py:705` | `# use the routing weights from the first token only for the entire sequence.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/modern/qcqa_attention.py:706` | `# This maintains O(1) KV mixing while preserving strict causality.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/qcqa_attention.py:716` | `# einsum: w_head[b,h_q,h_kv] · K_perm[b,h_kv,j,d] → (B, H_q, N_kv, d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/qcqa_attention.py:820` | `# Context: (B, H_q, L, N_kv) @ (B, H_q, N_kv, d) → (B, H_q, L, d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/vig_attention.py:505` | `# Frobenius norm verification (>= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/modern/vig_attention.py:651` | `# Branch 2: Global Linear Recurrence / Attention with denom.clamp(min=1.0)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/sparse/adaptive_sparse_attention.py:303` | `# Head dim fallback from source_attn if head_dim <= 0 or not resolved properly` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/adaptive_sparse_attention.py:758` | `# Apply RoPE (accept from position_embeddings or fallback to rotary_emb)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/sparse/adaptive_sparse_attention.py:773` | `# 2. Try rotary_module(Q, seq_len=...)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/sparse/bam_umoe.py:431` | `# If svd_rank >= min(dim, all_head_size), degenerates to dense (no compression)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/cluster_attention.py:749` | `# Apply RoPE (accept from position_embeddings or fallback to rotary_emb)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/sparse/cluster_attention.py:782` | `# Check for causal decoding (guard against self.config being None)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/sparse/expert_choice_moe.py:236` | `# Expert-Choice applies softmax over the sequence (dim=0), which catastrophically leaks future tokens into past states.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/expert_choice_moe.py:237` | `# We now apply softmax over the experts (dim=-1) and select top-k experts per token.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/expert_choice_moe.py:731` | `# Frobenius norm verification (>= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/moe_deltakv_paged.py:778` | `# Paged MoE block attention (executed when num_pages >= 2)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/moe_deltakv_paged.py:835` | `# Only bias page 0 for queries whose page index >= 0 (always true)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/moe_deltakv_paged.py:1043` | `# MDP-FIX-13: always return a 2-tuple with attn = tensor or None` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/sparse/routing_attention.py:712` | `# Sort + Gather + Chunk attention (real O(N²/K))` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/routing_attention.py:737` | `# → argsort descending puts cluster members first (ordered by position desc)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/routing_attention.py:756` | `# Compact attention: (B, H, max_q, max_k) — this is O((N/K)²), not O(N²)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/sparse/routing_attention.py:781` | `# Scatter back (each token belongs to exactly one cluster → safe)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/goldfinch_ahn.py:193` | `# Ablation: return v (no recurrence)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/ssm/goldfinch_ahn.py:233` | `# where g_chunk_j = g of last position in chunk j (approximation)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/goldfinch_ahn.py:241` | `# cum_log_omg[c] = sum_{j<=c} log(1-g_j)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/goldfinch_ahn.py:245` | `# We use a sequential scan over chunks only (n_chunks ~ 8 for L=128, chunk=16)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/goldfinch_ahn.py:253` | `# Full output: h_full[b, c, t] = h_carry[b, c] * prod_{t' <= t}(1-g) + h_local[b, c, t]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/goldfinch_ahn.py:254` | `# The h_local already includes the intra-chunk recurrence from zero state.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/goldfinch_ahn.py:256` | `#   h_full[c, t] = h_local[c, t] + h_carry[c] * prod_{t' <= t}(1 - g[c, t'])` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/goldfinch_ahn.py:391` | `# If GOLDFINCH_FULL_CONTEXT=1, window_radius = inf (full causal)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/goldfinch_ahn.py:841` | `# Try SDPA with bool mask (True = keep, False = mask)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/griffin_ssd.py:367` | `# decay: (1, H, 1, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:369` | `# log decay for stable exponentiation` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:391` | `# diff[i,j] = i - j (causal: i >= j)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:395` | `# D_intra[i,j] = decay^(i-j) if i>=j else 0` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:396` | `# log_decay shape: (1, H, 1, 1) → need (1, H, 1, 1, 1) for broadcasting` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:403` | `# q_c: (B,H,nc,C,d), k_c^T: (B,H,nc,d,C) → scores: (B,H,nc,C,C)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:411` | `# Each chunk's final state: state_after_chunk[c] = decay^C * state_before + sum of kv in chunk` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:412` | `# Contribution of chunk c to next chunks: k_v_chunk[c] = sum_{i=0}^{C-1} decay^(C-1-i) * k_c[i]^T @ v_c[i]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:413` | `# Actually simpler: state contribution = k_c^T @ v_c (matrix) with proper decay weights` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:415` | `# For chunk c, kv_contrib[c] = sum_{i=0}^{C-1} decay^(C-1-i) * (k_c[i]^T @ v_c[i])` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:416` | `# decay^j where j = C-1-i` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:418` | `# decay_weights[i] = decay^(C-1-i)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:424` | `# weighted k: (B,H,nc,C,d) * (1,H,1,C,1) → (B,H,nc,C,d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:428` | `# (B,H,nc,d,C) @ (B,H,nc,C,d) → (B,H,nc,d,d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:431` | `# Now parallel scan over chunks with chunk_decay = decay^chunk_size` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:437` | `# states_before[c] = chunk_decay * states_before[c-1] + kv_contrib[c-1]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:445` | `# y_inter[b,h,c,i] = q_c[b,h,c,i] @ states_before[b,h,c] * decay^(i+1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:446` | `# Actually: q at position i in chunk c sees state_before[c] with decay^(i+1)?` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:448` | `# state_before[c] = sum_{c'<c} decay^(chunk_size * (c-1-c')) * kv_contrib[c']` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:449` | `# For query at position i in chunk c, the contribution from past chunks is:` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/griffin_ssd.py:450` | `#   q_i @ state_before[c] * decay^(i) — the (i) accounts for positions within current chunk` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:453` | `# At position i in chunk c, we've had i decay steps since start of chunk` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:454` | `# state_at_position_i = decay^i * state_before[c] + sum_{j=0}^{i-1} decay^(i-1-j) * k[c,j]^T v[c,j]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:455` | `# The first term: query sees decay^i * state_before[c]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/griffin_ssd.py:459` | `# decay^i for i in [0, C-1]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/hymba_transx.py:473` | `# HX-FIX-01: alpha init = 4.0 → sigmoid(4.0) ≈ 0.98 (attention-dominant)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/hymba_transx.py:736` | `# HX-FIX-06: W_q2 and W_k2 = EXACT copies (no noise)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/hymba_transx.py:937` | `# alpha init = 4.0 → sigmoid(4.0) ≈ 0.98 (attention-dominant)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/jamba_info.py:382` | `# FIX 2: Per-head λ_init via Diff Transformer paper formula.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/jamba_info.py:383` | `# Formula: λ_init[h] = 0.8 - 0.6 · exp(-0.3 · h)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/jamba_info.py:939` | `# Determine sequence dim: for (B, L) it's dim=1;` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/jamba_info.py:940` | `# for (B, 1, L, L) it's dim=-1; for (B, L, L) it's dim=-1` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/mamba_block.py:184` | `# FIXED: padding=self.d_conv - 1 causes right-padding, leaking future tokens.` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/ssm/mamba_block.py:185` | `# Changed to padding=0; manual left-padding is now applied during forward pass.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/mamba_block.py:391` | `# x_ssm path = rows [0 : d_inner]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_block.py:392` | `# z     path = rows [d_inner : 2*d_inner]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_block.py:396` | `# x_ssm: full projection from W_v` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/ssm/mamba_block.py:402` | `# rows [0 : dt_rank]                     → dt  (keep init)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_block.py:403` | `# rows [dt_rank : dt_rank + d_state]     → B   (from W_k)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_block.py:404` | `# rows [dt_rank + d_state : ...]         → C   (from W_q)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_block.py:432` | `# shape (d_inner, 1, d_conv)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_block.py:465` | `# past_len from conv_state if available` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/mamba_block.py:666` | `# conv_prev: (B, d_inner, d_conv - 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/mamba_svd_flash.py:176` | `# Outputs from the bottleneck:` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/nexus.py:1047` | `# PATH 1: CAUSAL (decoder) → Chunked Causal NEXUS` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/nexus.py:1059` | `# PATH 2: CROSS-ATTENTION or SHORT SEQUENCE → dense` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/nexus.py:1089` | `# PATH 3: BIDIRECTIONAL (encoder) → sparse NEXUS` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/orion.py:763` | `# Scale x ONCE by 1/sqrt(d). This is the "x" entering the FAVOR+ formula.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/orion.py:772` | `# Subtract \|\|x\|\|²/2 (per query/key) → unbiased FAVOR+ feature map` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/phantom.py:86` | `# Masked positions (future tokens) are <= -1000.0. Preserve mask while clamping unmasked values.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/phantom.py:185` | `# Ill-conditioned CUR intersection -> fallback to stable truncated SVD` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/samba_moe.py:286` | `# Nested encoder: bypass outer LayerNorm (self.norm) and double residual` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/selective_ssm.py:83` | `# Input projection: x -> (x_ssm, z)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/selective_ssm.py:99` | `# Delta projection from dt_rank -> d_inner` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/selective_ssm.py:220` | `# FIX: Prevent State Poisoning from Padded Tokens` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/ssm/simple_ssm.py:139` | `# Recurrence decay gate` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/simple_ssm.py:272` | `# Replace O(L^2) dense matrix with O(L) recurrent scan` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/spectral_mamba.py:399` | `# With gate_bias=10.0, sigmoid(10.0) ≈ 0.99995, so the model starts as ~pure attention (identical to pretrained).` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/spectral_mamba.py:593` | `# For GQA: mean-average the head groups from source` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/spectral_mamba.py:829` | `# Support both `layer=` and `model=` call conventions` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:225` | `# In our scan, we compute A = +exp(A_log) (positive),` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:226` | `# then log_decay = -dt * A (negative), giving decay < 1.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:228` | `# A = -exp(A_log), decay = exp(dt * A).` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:235` | `# Mamba paper: dt_rank = ceil(d_model / 16), not d_inner` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:238` | `# x_proj maps inner_dim → dt_rank + 2*d_state` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py:246` | `# dt_proj maps dt_rank → inner_dim` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/ssm/vspec.py:127` | `# Initialize head gate to neutral 1.0 (bias=3.0 -> sigmoid(3.0) ≈ 0.95)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/asvd_attention.py:694` | `# 3. RoBERTa / BERT Nested Encoder: self.query, self.key, self.value` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/svd/asvd_attention.py:954` | `# FIXED: Keyword Argument Cleanup: pop rank and copy_weights early from kwargs` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:643` | `# Weight verification (Frobenius norm >= 1e-4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:746` | `# num_inter = (qc @ S) * decay_from_prev` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:747` | `# qc: (B, H, C, d), S: (B, H, d, d) -> output: (B, H, C, d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:751` | `# denom_inter = (qc @ z) * decay_from_prev` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:752` | `# qc: (B, H, C, d), z: (B, H, d, 1) -> output: (B, H, C, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:766` | `# MEMORY FIX: Apply decay to keys BEFORE outer product!` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:771` | `# chunk_S = sum_{c} (kc_decayed_c \otimes vc_c)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/factorised_matrix_state.py:774` | `# chunk_z = sum_{c} kc_decayed_c` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/fused_svd_kernel.py:74` | `# Step 1: Compute SRAM-resident Z = X @ V^T of shape [BLOCK_M, R]` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/svd/fused_svd_kernel.py:111` | `# Step 3: Up-projection Y = Z_norm @ U^T + bias: [BLOCK_M, R] @ [R, BLOCK_K_OUT]` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/svd/fused_svd_kernel.py:988` | `# Frobenius norm verification >= 1e-4 on copied weights` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/lora_attention.py:681` | `# FIXED: Removed destructive torch.nan_to_num(..., nan=-1e4) before softmax` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/lora_attention.py:796` | `# `rank` from sweep is ignored.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/projection_factorized_attn.py:44` | `# they are positional/keyword arguments of the delegate signature itself.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/svd/salt_attention.py:592` | `# Recalculate adaptive ranks from actual extracted weights if available` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/svd/svd_attention.py:292` | `# Persistent SVD projection buffer (populated in copy_weights from W_K)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/svd/svd_attention.py:500` | `# GQA: take the first group from each cluster (deterministic)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/svd/svd_attention.py:549` | `# W_K shape: [kv_head_size, hidden_size] → SVD gives right singular vectors of shape [hidden_size, ...]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_attention.py:557` | `# svd_proj: [hidden_size, r] — project hidden_states → rank-r space` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_attention.py:635` | `# (SVD with r >= N is meaningless — the projection is identity)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_attention.py:718` | `# Use the fixed SVD subspace precomputed from W_K in copy_weights.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/svd/svd_attention.py:726` | `# kv_input: [B, kv_len, hidden_size] → scores: [B, kv_len, actual_r]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_linear_attention.py:226` | `# (B, 1, 1, N) -> (B, 1, N, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_linear_attention.py:229` | `# (B, H, N, N) -> take diagonal` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_linear_attention.py:970` | `# decay` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_linear_attention.py:985` | `# padded positions do not decay` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/svd_mamba.py:744` | `# 2. Selective SSM Path (Bounded FP32 associative update with min=1.0 denominator clamp)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/svd/svd_projection_attention.py:623` | `# Build attention mask for SDPA (bool mask where True = keep)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/svd/svd_projection_attention.py:627` | `# (B, N) → (B, 1, 1, N) bool` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/svd/triton_svd_linear_attention.py:358` | `# 1. First Pass: S = K^T V and Z = sum(K)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/benchmarks/adapters/zeroshot.py:90` | `# Build prompt samples (cached from datasets or generated for evaluation)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/benchmarks/core/model_loader.py:200` | `# Handle constructor signatures: some take (model, rank=...), some take (rank=...)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/benchmarks/executor.py:151` | `# 1. Inspect module metadata from discovery (or handle baseline)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/__init__.py:76` | `# 2. Import all subpackages and re-export classes at top level` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/experimental/hashed_net_compressor.py:506` | `# compression_ratio ∈ (0, 1]: n_buckets = total_w · ratio  → smaller = more compressed` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/experimental/liftr.py:118` | `# GPT-2 and BERT-style return tuples.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/experimental/liftr.py:243` | `# Cannot call — return an empty tensor of the right shape.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py:221` | `# Reconstruction cache: layer_idx -> (full_k, full_v, n_entries_seen)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py:470` | `# Infer dtype from the first entry's sink or quantized buffer` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/come_compressor.py:60` | `# Conservative mapping — higher rank => more aggressive shedding.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/come_compressor.py:814` | `# Prefer explicit rank, otherwise derive from sparsity` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/come_compressor.py:1026` | `# derive from rank if possible, else let __init__ decide` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/compot_compressor.py:355` | `# Inspect layer dimensions to distinguish dense MLP projections (e.g. 768->3072, 3072->768)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/compot_compressor.py:356` | `# from standard attention projections. Wide MLP layers destroy up to 93% of the Frobenius norm` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/cr_compressor.py:112` | `# Conv1D weights are [in_features, out_features] -> transpose to [out_features, in_features]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/cur_compressor.py:166` | `# (..., in_features) -> (..., rank)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/cur_compressor.py:168` | `# (..., rank) -> (..., rank)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/cur_compressor.py:170` | `# (..., rank) -> (..., out_features)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/frandi_compressor.py:182` | `# Merge into B: B_new = B + (alpha/r) * (lora_B @ lora_A) @ pinv(A)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/frandi_compressor.py:183` | `# Simpler: absorb into B by solving B_new A = B A + delta_BA` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/low_rank/frandi_compressor.py:187` | `# Solve for B_new such that B_new @ A = B @ A + delta:` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/frandi_compressor.py:282` | `# FR-FIX-10: solve for A such that A @ L = A_tilde.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/frandi_compressor.py:283` | `# Transposed: L^T @ A^T = A_tilde^T. L^T is upper triangular.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/mlorq_compressor.py:220` | `# (B, in) @ (in, r) -> (B, r)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/mlorq_compressor.py:521` | `# Rank list (unique, <= max_rank)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/modegpt_compressor.py:289` | `# MD-FIX-09: empty list → None` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/modegpt_compressor.py:559` | `# MD-FIX-13: LoRA rank inherits from wrapper if present` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/nmf_compressor.py:124` | `# Sign-aware reconstruction: y = x @ W^T where W = H@V * W_sign.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/nmf_compressor.py:175` | `# rank*(in+out) < in*out  →  rank < in*out / (in+out)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/nmf_compressor.py:186` | `# NMF-FIX-05: compute on GPU if available, return to orig device.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/slim_compressor.py:116` | `# Fix 2: Reliable Conv1D check using class name and hasattr(nf)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/low_rank/slim_compressor.py:151` | `# Fix 2: Check for Conv1D via class name and nf attribute` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/low_rank/soka_compressor.py:59` | `# V: [in_features, rank] -> maps x to bottleneck` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/soka_compressor.py:60` | `# U: [out_features, rank] -> maps bottleneck to output` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/soka_compressor.py:228` | `# U: [d_out, r], V: [d_in, r]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/low_rank/svd_llm_compressor.py:114` | `# LoRA for W_u: input dim = r, output dim = out_features` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/svd_llm_compressor.py:122` | `# LoRA for W_v: input dim = in_features, output dim = r` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/low_rank/tomoe_compressor.py:64` | `# Conv1D stores weight as (nx, nf) where nx=in_features, nf=out_features` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/pruning/armor_pruner.py:655` | `# AR-FIX-01: alpha defaults to rank so scaling = 1.0.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/pruning/ast_pruner.py:93` | `# FIXED: Ensure base weight is trainable with requires_grad=True so STE passes dense gradients back` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/pruning/ast_pruner.py:544` | `# W: (d_out, d_in)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/pruning/ast_pruner.py:604` | `# Base weights remain trainable (requires_grad=True) to absorb KD gradients.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/pruning/bonsai_pruner.py:108` | `# FIXED: Mask starts at all ones (0% initial sparsity = 0 initial damage)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/pruning/bonsai_pruner.py:148` | `# Gradually ramp sparsity from 0.0 to target_sp via cubic warmup schedule.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/pruning/bonsai_pruner.py:450` | `# Base weights start 100% dense (0% damage) and remain trainable (requires_grad=True).` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/pruning/bonsai_pruner.py:451` | `# Sparsity increases smoothly over self.target_steps in forward passes.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/pruning/mamba_shedder.py:620` | `# 2:4 → no storage savings (dense storage kept). Fallback to` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/pruning/mamba_shedder.py:659` | `# Channel pruning → real storage savings.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/coopq_quantizer.py:424` | `# Allowed bit-levels with floor >= self.min_bits (>= 4)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:89` | `# (..., 1) vs (1, ..., C) → (..., C)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:147` | `# x: (..., D_in)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:347` | `# P_r as a (D_in, r) selector matrix via scatter` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:392` | `# Use a single shared centroid table = average of per-column tables.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:408` | `# Dual-mode: .compress(W, X) → tuple, .compress() → (model, stats)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:444` | `# Synthesize a small sample matrix from the mean vector.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/slicegpt.py:199` | `# Flatten leading dims: (B, seq, d) -> (B*seq, d)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/slicegpt.py:203` | `# Online covariance: C += X^T X` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/slicegpt.py:394` | `# Rank r cannot exceed min(d_out, d_in)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/slicegpt_pca.py:609` | `# Rank cannot exceed min(d_out, d_in)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/slicegpt_pca.py:653` | `#   U_factor  = W_pca[:, :keep]      shape (d_out, keep)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/quantization/slicegpt_pca.py:654` | `#   Vt_factor = V[:, :keep].T        shape (keep, d_in)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/adasvd_compressor.py:1202` | `# FIXED: Adaptive Rank Strategy: Force rank=64 (or 128 for fused above)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/direct_svd.py:383` | `# Handle freezing of non-target layers when target_attention_only=True` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/dobi_svd.py:440` | `# If target_attention_only=True, freeze non-target linear/conv layers during recovery` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/dynamic_rank.py:155` | `# Parameter cost for layer l at rank r is r * (d_in + d_out)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/dynamic_rank.py:231` | `# Greedily subtract ranks from layers with lowest entropy that are above per_layer_min` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/dynamic_rank.py:366` | `# This serves as both the default path (when covariance=None) and the` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/dynamic_rank.py:380` | `# RESEARCH FIX: Activation-aware whitening SVD with eigenvalue clamping lambda_i >= 1e-4` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/dynamic_rank.py:383` | `# covariances (e.g. from dummy calibration data) that inflate inv_sqrt by up to 100x` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/dynamic_rank.py:387` | `# Eigen-decomposition with regularized eigenvalue clamping >= 1e-4` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/dynamic_rank.py:634` | `# (rank <= num_dummy_tokens << hidden_dim), causing the whitening transform` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/dynamic_rank.py:706` | `# 2x32 random tokens) produce a rank-deficient covariance (rank<=64 for` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/dynamic_rank.py:707` | `# hidden_dim=768). The whitening eigenvalue clamp (min=1e-4) then assigns` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/dynamic_rank.py:712` | `# Solution: return {} so from_linear uses plain weight-only SVD instead.` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/svd/dynamic_rank.py:797` | `# Freeze uncompressed layers when target_attention_only=True` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/dynamic_rank.py:944` | `# Register classes under *_class to avoid collision with factory functions` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/optimal_subspace_projection.py:496` | `# OSP-FIX-10: empty rank → fallback.` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/residual_svd_compressor.py:222` | `# from the residual tail E = W - U_r V_r minimizes initial reconstruction loss to zero-shot optimality.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/residual_svd_compressor.py:415` | `# Alias class for PEFT terminology` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/svd/saes_svd_v2.py:112` | `# Fix 3: Reliable Conv1D check using class name and hasattr(nf)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/svd/saes_svd_v2.py:136` | `# Fix 3: Conv1D detection by class name or nf attribute` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/compression_engine/svd/swift_svd_act.py:118` | `# SW-FIX-12: empty list → None` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/swift_svd_act.py:234` | `# Single tensor / dict / small tuple → single batch` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/swift_svd_act.py:250` | `# Distinguish "list of batches" from "single batch as tuple".` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/svd/swift_svd_act.py:316` | `# Shape changed across batches → skip` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/svd/swift_svd_act.py:431` | `# SW-FIX-13: monotone selection — start from r_lower, aim for r_upper.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py:256` | `# Step 1: contract with B over n2  ->  (K * n1, m2)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py:261` | `# Step 2: (K, m2, n1) -> (K * m2, n1) @ A^T  ->  (K * m2, m1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/compression_engine/tensor_decomp/tt_compressor.py:400` | `# Adaptive rank: smallest r with retained energy >= threshold` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/_attention_extractors.py:91` | `# Infer from q_proj and head_dim (Qwen, LLaMA)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/_attention_injectors.py:329` | `# Encoder self-attention without internal out_proj -> initialize to identity` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/architecture.py:126` | `# If module A's path is a strict prefix of module B's path (e.g. A="...attention", B="...attention.self"),` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/autograd/flash_attention.py:48` | `# Recompute P in backward from saved LSE (saving SRAM/HBM footprint)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/autograd/flash_attention.py:63` | `# dS = P * (dP - D) where D = sum(grad_out * out, dim=-1, keepdim=True)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/autograd/flash_attention.py:64` | `# Using standard softmax backward: dS = P * (dP - (P * dP).sum(-1, keepdim=True))` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/common.py:113` | `# (N, D) → (1, 1, N, D) to broadcast against (B, H, N, D)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/common.py:453` | `# so lin1 = S_r^{1/2} Vh_r and lin2 = U_r S_r^{1/2}.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/model_family.py:54` | `# Bidirectional encoder families (for mask_rank=2).` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/pruning_utils.py:144` | `# Flatten batch and sequence dimensions: [B*L, d_in]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/pruning_utils.py:170` | `# Multi-tier inversion strategy: Cholesky -> Standard LU Inv -> Moore-Penrose Pseudo-Inv` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/registry.py:145` | `# Derive capabilities from explicit kwargs or attributes on the registered object` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/selective_scan.py:161` | `# Cumulative log-decay within the chunk` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/_common.py:131` | `# [B, N, D] -> [B, 1, N, D]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/cosformer_triton_kernel.py:58` | `# محاسبهٔ score = Q_cos @ K_cos^T + Q_sin @ K_sin^T  →  (BLOCK_N, BLOCK_N)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:77` | `# Load constant decay for this head: decay_h in (0, 1)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:116` | `# S_t = decay_val * S_{t-1} + k_f^T @ v_f (outer product: [i, j] = k_f[i] * v_f[j])` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:120` | `# Z_t = decay_val * Z_{t-1} + k_f` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:234` | `# Accumulator for decay gradient for this head from this batch item` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:317` | `# Gradient through ELU + 1: dELU(x)/dx = 1 if x > 0 else exp(x)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/triton_kernels/fms_scan.py:321` | `# dv_t from S_t = decay * S_{t-1} + k_f^T @ v_f:` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:326` | `# dk_f from S_t and Z_t:` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/triton_kernels/fms_scan.py:353` | `# Decay gradient accumulation:` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:354` | `# S_t = decay * S_{t-1} + ... => dL/ddecay += sum(dS * S_{t-1})` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:355` | `# Z_t = decay * Z_{t-1} + ... => dL/ddecay += sum(dZ * Z_{t-1})` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:379` | `# dS_{t-1} = decay * dS_t, dZ_{t-1} = decay * dZ_t` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:383` | `# Atomically accumulate ddecay into global head buffer` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/fms_scan.py:533` | `# Ensure decay is (H,)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/gdn_scan.py:104` | `# k_t is (D,) and S is (D, D). Result = sum_i k_t[i] * S[i, :]` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/triton_kernels/gdn_scan.py:248` | `# Load S_t and S_{t-1} from workspace` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/triton_kernels/gdn_scan.py:299` | `# Path through norm_sq in step = alpha_t / denom:` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/core/triton_kernels/gla_scan.py:269` | `# If denom_raw <= 1e-4, clamp_min is flat => gradient through denom_raw is 0` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/low_rank.py:97` | `# V_T has shape [D_in, Rank]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/low_rank.py:101` | `# U_T has shape [Rank, D_out]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/low_rank.py:170` | `# V_mat has shape [D_in, Rank]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/low_rank.py:174` | `# U_mat has shape [Rank, D_out]` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/sliding_window.py:83` | `# Causal sliding window mask: (i - j >= 0) and (i - j < window_size)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/core/triton_kernels/state_update.py:100` | `# Discretize: dA = exp(delta * A), dB = delta * B` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/core/triton_kernels/state_update.py:107` | `# Output: y = sum(h * C) + D * u` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/evaluation/label_quality.py:106` | `# ── Confident Joint (core algorithm) ────────────────────────────` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/experiment_utils/tasks.py:178` | `# Loss plot from training history (if available)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/inference/paged_cache.py:40` | `# Maps sequence_id -> list of block indices` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/inference/paged_cache.py:42` | `# Maps sequence_id -> current token length` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/runners/run_autonomous_48h_pipeline.py:510` | `# STREAM 5: Stage 5 — Long-Context Sequence Length Scaling: N=128..1024 (Batch 32-64)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/runners/run_autonomous_48h_pipeline.py:629` | `# Launch parallel workers pulling from the shared pool of remaining tasks` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/trainer/__init__.py:15` | `# Import trainers to register them` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/trainer/hf_trainer.py:76` | `# Disable safetensors to avoid RuntimeError from shared-memory tensors in custom attention modules` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/trainer/utils.py:101` | `# Shape-based heuristic for low-rank bottleneck: tall = U, wide = V` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/utils/replace_attention.py:75` | `# this injected call, never from the base model.` | `KEEP` | 0.85 | Explanatory English comment |
| `scripts/normalize_results.py:82` | `# Override metadata from explicit top-level dict fields if present` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:39` | `# 1. RANKING HELPERS` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:58` | `# Use average ranking for ties` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:74` | `# Compute mean performance per dataset` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:76` | `# Subtract row mean (align)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:78` | `# Rank all aligned values together` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:87` | `# 2. OMNIBUS TESTS (multiple algorithms, multiple datasets)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:99` | `# Iman-Davenport correction (recommended over plain Friedman)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:131` | `# Expected rank sum under H0` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:159` | `# Step 1: Compute range per dataset (row)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:161` | `# Handle zero-range rows` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:164` | `# Step 2: Rank the ranges` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:167` | `# Step 3: Compute ranks within each row, then weight by range rank` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:179` | `# Step 4: Total sum of squares` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:182` | `# Step 5: Test statistic` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:195` | `# 3. POST-HOC PROCEDURES` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:347` | `# Standard error of rank difference` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:361` | `# Number of simultaneous hypotheses` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:387` | `# 4. PAIRWISE TESTS (two algorithms)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:400` | `# Classification of effect (Demšar 2006 heuristic)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:465` | `# 5. EFFECT SIZES` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:477` | `# Use standard deviation of differences` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:528` | `# Interpretation: \|delta\| < 0.147 = negligible, < 0.33 = small, < 0.474 = medium, ≥ 0.474 = large` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:585` | `# 6. BAYESIAN TESTS (Benavoli et al. 2017)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:613` | `# Standard error corrected for correlation in CV (Nadeau & Bengio, 2003)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:615` | `# For k-fold CV, the correction factor depends on the number of folds.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:616` | `# A common approximation for 10‑fold CV is n_test / n_train ≈ 1/9.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:657` | `# Exclude zeros for rank computation` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:663` | `# Signed rank sum (Wilcoxon style)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/comparison/statistical_tests.py:668` | `# Dirichlet posterior for the probability that a positive difference occurs` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/comparison/statistical_tests.py:670` | `# Prior: symmetric Dirichlet` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/comparison/statistical_tests.py:700` | `# 7. CONVENIENCE WRAPPER (multi‑comparison)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:308` | `# Compute and plot Pareto frontier (maximizing accuracy, minimizing time)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:651` | `# Save enriched dataframe to JSON` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/layerwise_analyzer.py:48` | `# FIX: Compute outlier and spectral metrics dynamically per layer and immediately free activations` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/layerwise_analyzer.py:52` | `# Flatten to [N, D]` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/layerwise_analyzer.py:67` | `# 1. Update running mean representation vector (CPU numpy)` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:76` | `# 2. Update Peak Outlier Magnitude` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/layerwise_analyzer.py:80` | `# 3. Update Outlier Ratio & Kurtosis Moments` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:194` | `# Filter to only the top-level block (e.g. layers.0, not layers.0.attention)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/layerwise_analyzer.py:220` | `# Extract tensor` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/layerwise_analyzer.py:223` | `# FIX: Update streaming accumulator dynamically and release activation immediately` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/layerwise_analyzer.py:289` | `# Extract per-layer mean vectors` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:316` | `# Estimate intrinsic dimension using mean vectors` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/layerwise_analyzer.py:364` | `# Maximum likelihood estimation of intrinsic dimension` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/module_analysis/main.py:51` | `# 1. Load registries` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/module_analysis/main.py:58` | `# Build entity to registered names mapping` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/module_analysis/main.py:73` | `# 2. Load CSV data` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/module_analysis/main.py:77` | `# 3. Load verification json` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/module_analysis/main.py:85` | `# 4. Enumerate 217 paper directories` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/module_analysis/parsers.py:98` | `# Check inside paper folder for python file` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/analysis/plotting.py:297` | `# 1. Accuracy vs Rank by model` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/plotting.py:305` | `# 2. Accuracy vs Rank by task` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/analysis/plotting.py:313` | `# 3. Speedup factor` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/plotting.py:331` | `# 4. Parameter reduction` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/plotting.py:353` | `# 5. Accuracy drop distribution` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/plotting.py:371` | `# 6. Memory reduction` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/plotting.py:392` | `# 7. Accuracy by task type` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/plotting.py:403` | `# 8. Compression ratio heatmap` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/analysis/plotting.py:418` | `# 9. Best config per dataset` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/__init__.py:77` | `# Renamed module alias mappings: (subpath, {old_class_name: target_attr_name})` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/adaptive_span_attention.py:874` | `# FIXED: Enforce min_span floor to avoid auto-regressive Markov bigram collapse ("eseseses...")` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:304` | `# Multi-Head Attention Projections equipped with trainable residual LoRA adapters` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:352` | `# Compatibility properties for parameter inspection & framework integration` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:414` | `# 1. Detect nested encoder architectures (RoBERTa / BERT)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:441` | `# Inherit rotary embeddings if present` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:452` | `# 2. GPT-2: Fused c_attn + c_proj` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:512` | `# 3. RoBERTa / BERT nested self-attention (pretrained_attn.self)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:530` | `# 4. Standard linear projections (DistilBERT, Qwen, LLaMA)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:549` | `# 5. Fallback via UniversalAdapter` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:567` | `# Verify all mandatory projections were extracted` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:579` | `# 6. Copy weights into target LoRALinear layers with strict dimension safety` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:592` | `# Inner nested self-attention bypasses out_proj; initialize to identity` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:601` | `# Handle transposed orientation if necessary (only if shape does not match already)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:734` | `# Subsequent tokens (only for bidirectional non-causal encoders)` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:755` | `# Guarantee CLS block inclusion for block_idx > 0` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:794` | `# 1. Base Projections` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:809` | `# 2. RoPE Embeddings applied before GQA expansion` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:836` | `# 3. GQA Expansion` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:841` | `# 4. Safe head mask reshaping` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:865` | `# Memory-efficient: use SDPA to avoid OOM on large seqs` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:896` | `# If a 4D attention mask is provided, it already incorporates causal + padding constraints` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:903` | `# Causal mask for decoders` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:925` | `# Hierarchical Intra-Block and Inter-Block Attention` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:956` | `# 5. Output projection with nested encoder bypass` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/block_transformer_hiba.py:958` | `# Outer RobertaAttention.output.dense performs the dense projection; bypass out_proj` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/eat.py:45` | `# Slice time dimension if mismatched` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/eat.py:498` | `# 1. GPT-2 fused c_attn` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/eat.py:566` | `# 2. Nested (BERT / RoBERTa)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/eat.py:585` | `# 3. Flat modules (DistilBERT, Qwen, LLaMA)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/eat.py:605` | `# UniversalAdapter fallback` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/eat.py:871` | `# EAT-FIX-03: is_nested_encoder is only True for truly nested containers` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/infini_attention.py:163` | `# Zero initialization for output projection to safely inject Mamba (ReZero)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/infini_attention.py:226` | `# Cache I/O helpers` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/infini_attention.py:353` | `# FIX 2: Pre-LayerNorms for Parallel Branches` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/infini_attention.py:357` | `# FIX 3: ReZero Scaling Parameters to ensure safe zero-shot KD initialization` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/infini_attention.py:529` | `# FIX 1 & 2: Parallel Pre-LN Delta Logic` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/infini_attention.py:530` | `# The module computes Parallel Deltas without re-adding the residual internally.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/lola_attention.py:68` | `# Recent cache` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/lola_attention.py:88` | `# Sparse cache (sample every 4th token)` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/lola_attention.py:111` | `# Low-rank subspace cache` | `KEEP` | 0.95 | Math formula, algorithm pseudocode, or theoretical derivation |
| `neuralops/attentions/adaptive/lola_attention.py:738` | `# LL-FIX-01/02/03: cache only for streaming decoders with past_key_value.` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/lola_attention.py:739` | `# For encoders (bidirectional), the cache would mix tokens across batches` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/lola_attention.py:740` | `# and hold large view-retained buffers, causing ~70% VRAM bloat.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/lola_attention.py:748` | `# Ensure stale state is dropped so batches stay isolated.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:324` | `# Aliases` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:410` | `# 1. GPT-2 fused` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:459` | `# 2. RoBERTa nested` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:476` | `# 3. Flat modules` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:608` | `# PTP-FIX-11: only force [CLS] if position 0 is valid.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:611` | `# Force first token to top priority (only if valid).` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:620` | `# FIXED: Global top-k pruning over the sequence axis violates causality.` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:621` | `# Disable pruning for auto-regressive generation/causal models.` | `KEEP` | 0.85 | Explanatory English comment |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:639` | `# Tuple cache` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:643` | `# HF DynamicCache` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:661` | `# Note: PTP prunes tokens ONLY for the current chunk (new tokens).` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:662` | `# With KV-cache, past tokens are always used in full.` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
| `neuralops/attentions/adaptive/progressive_token_pruning.py:670` | `# Concatenate past (full) with current (pruned).` | `KEEP` | 0.98 | Docstring-like explanation or architectural rationale |
