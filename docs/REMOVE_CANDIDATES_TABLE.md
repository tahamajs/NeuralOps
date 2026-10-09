# REMOVE Candidates Table (Part 1.2)

Total candidates: 37

| # | File | Line | Commented Code | Reason |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `neuralops/core/pruning_utils.py` | 161 | `# H = (2/N) * X^T X` | Commented variable assignment |
| 2 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 658 | `# past_len = N_kv - N_q is exact for both prefill and decode-with-cache.` | Commented variable assignment |
| 3 | `neuralops/attentions/linear/nystrom_attention.py` | 646 | `# F1 = softmax(Q_c @ K_landmarks^T)  → (B, H, C_len, m_prev)` | Commented variable assignment |
| 4 | `neuralops/attentions/linear/nystrom_attention.py` | 650 | `# F2 = softmax(Q_landmarks @ K_landmarks^T)  → (B, H, m_prev, m_prev)` | Commented variable assignment |
| 5 | `neuralops/attentions/linear/nystrom_attention.py` | 663 | `# F3 = softmax(Q_landmarks @ K_prev^T)  → (B, H, m_prev, start)` | Commented variable assignment |
| 6 | `neuralops/attentions/ssm/goldfinch_ahn.py` | 263 | `# h_full = h_local + h_carry * prod_local` | Commented variable assignment |
| 7 | `neuralops/attentions/ssm/spectral_mamba.py` | 398 | `# output = gate * attn + (1 - gate) * mamba` | Commented variable assignment |
| 8 | `neuralops/compression_engine/low_rank/come_compressor.py` | 113 | `# residual = B @ (A @ t^T)^T + bias` | Commented variable assignment |
| 9 | `neuralops/compression_engine/low_rank/frandi_compressor.py` | 188 | `# B_new = B + delta @ A^+` | Commented variable assignment |
| 10 | `neuralops/compression_engine/low_rank/modegpt_compressor.py` | 422 | `# U_r = U[:, :r] · diag(S[:r]);  V_r = Vt[:r, :]` | Commented variable assignment |
| 11 | `neuralops/compression_engine/low_rank/svd_llm_compressor.py` | 204 | `# latent = x @ W_v.T + (optional LoRA_v)` | Commented variable assignment |
| 12 | `neuralops/compression_engine/low_rank/svd_llm_compressor.py` | 214 | `# out = latent @ W_u.T + (optional LoRA_u)` | Commented variable assignment |
| 13 | `neuralops/compression_engine/quantization/slicegpt.py` | 423 | `# W_reconstructed = W_pca[:, :keep] @ Q[:, :keep]^T` | Commented variable assignment |
| 14 | `neuralops/compression_engine/quantization/slicegpt_pca.py` | 651 | `# W_reconstructed = W_pca[:, :keep] @ V[:, :keep].T` | Commented variable assignment |
| 15 | `neuralops/compression_engine/svd/erc_svd.py` | 408 | `# fallback_to_original=False: proceed with high error (warned).` | Commented variable assignment |
| 16 | `neuralops/compression_engine/svd/swift_svd_act.py` | 470 | `# M = U_act^T W V_act` | Commented variable assignment |
| 17 | `neuralops/core/autograd/flash_attention.py` | 57 | `# dV = P^T * grad_out` | Commented variable assignment |
| 18 | `neuralops/core/autograd/flash_attention.py` | 60 | `# dP = grad_out * V^T` | Commented variable assignment |
| 19 | `neuralops/core/autograd/flash_attention.py` | 68 | `# dQ = dS * K` | Commented variable assignment |
| 20 | `neuralops/core/autograd/flash_attention.py` | 71 | `# dK = dS^T * Q` | Commented variable assignment |
| 21 | `neuralops/core/autograd/fused_backward.py` | 44 | `# P = softmax(scores)` | Commented variable assignment |
| 22 | `neuralops/core/autograd/fused_backward.py` | 47 | `# dV = P^T @ dO` | Commented variable assignment |
| 23 | `neuralops/core/autograd/fused_backward.py` | 50 | `# dP = dO @ V^T` | Commented variable assignment |
| 24 | `neuralops/core/autograd/fused_backward.py` | 53 | `# dS = P * (dP - sum(dP * P))` | Commented variable assignment |
| 25 | `neuralops/core/autograd/fused_backward.py` | 58 | `# dQ = dS @ K` | Commented variable assignment |
| 26 | `neuralops/core/autograd/fused_backward.py` | 61 | `# dK = dS^T @ Q` | Commented variable assignment |
| 27 | `neuralops/core/triton_kernels/fms_scan.py` | 124 | `# num = q_f @ S -> num[j] = sum_i q_f[i] * S[i, j]` | Commented variable assignment |
| 28 | `neuralops/core/triton_kernels/fms_scan.py` | 126 | `# denom = q_f @ Z -> scalar` | Commented variable assignment |
| 29 | `neuralops/core/triton_kernels/fms_scan.py` | 305 | `# num_t = q_f @ S_t  =>  dL/dS_t += q_f^T @ dnum_t` | Commented variable assignment |
| 30 | `neuralops/core/triton_kernels/fms_scan.py` | 307 | `# denom_t = q_f @ Z_t => dL/dZ_t += q_f * ddenom_t` | Commented variable assignment |
| 31 | `neuralops/core/triton_kernels/gdn_scan.py` | 103 | `# inter_v = k_t @ S -> (D,)` | Commented variable assignment |
| 32 | `neuralops/core/triton_kernels/gdn_scan.py` | 118 | `# y_t = q_t @ S -> (D,)` | Commented variable assignment |
| 33 | `neuralops/core/triton_kernels/gdn_scan.py` | 313 | `# dS_prev = beta_t * dS - step * k_t @ k_t^T @ dS` | Commented variable assignment |
| 34 | `neuralops/core/triton_kernels/gla_scan.py` | 102 | `# S_t = lam_t * S_{t-1} + k_t^T @ v_t` | Commented variable assignment |
| 35 | `neuralops/core/triton_kernels/gla_scan.py` | 106 | `# Z_t = lam_t * Z_{t-1} + k_t` | Commented variable assignment |
| 36 | `neuralops/core/triton_kernels/low_rank.py` | 188 | `# dz = grad_out @ U_mat.t() -> [M, Rank]` | Commented variable assignment |
| 37 | `neuralops/core/triton_kernels/low_rank.py` | 196 | `# z = x @ V_mat -> [M, Rank]` | Commented variable assignment |
