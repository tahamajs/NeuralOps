# d031 fused_svd_kernel  [attention]
aliases: fused_svd_kernel
file: Codes/framework/attentions/fused_svd_kernel.py  doc: Docs/content/attentions/fused_svd_kernel.md
doc: Fused SVD-Linear Layer with SRAM-Resident RMSNorm & Dual-Projection. | RESEARCH FIX: Eliminates GPU High-Bandwidth Memory (HBM) roundtrip latency by fusing: | 1. Down-projection: Z = X @ V^T  (hidden_dim -> rank) | 2. In-SRAM RMSNorm: Z_norm = Z * rsqrt(mean(Z^2) + eps) * weight
class _FusedSVDLinearAutogradFunction(Function)
  doc: Autograd Function dispatching Triton JIT fused kernel on CUDA GPUs | with exact analytical backward gradient computation.
class FusedSVDLinear(Module)
  doc: RESEARCH SOTA: Fused SVD-Linear Module. | Combines low-rank projection $Z = x V^T$, in-SRAM RMSNorm, and up-projection $Y = Z U^T + b$
forward(x, V, U, norm_weight, bias)
__init__(in_features, out_features, rank, bias, use_rms_norm, eps, dtype, device)
forward(x)
tensor_ops:
  x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
  return y_2d.view(*orig_shape[:-1], K_out)
  return y_2d.view(*orig_shape[:-1], K_out)
  grad_output_2d = grad_output.reshape(-1, U.shape[0]).contiguous()
  grad_U = torch.matmul(grad_output_2d.T, z_norm_c).to(U.dtype)  # [K_out, R]
  grad_z_norm = torch.matmul(grad_output_2d, U_c)  # [M, R]
  grad_V = torch.matmul(grad_z.T, x_2d_c).to(V.dtype)  # [R, K_in]
  grad_x = torch.matmul(grad_z, V_c).view(*grad_output.shape[:-1], x_2d.shape[-1]).to(x_2d.dtype)
parameters: torch.empty(rank, in_features, dtype=dtype, device=device | torch.empty(out_features, rank, dtype=dtype, device=device | torch.ones(rank, dtype=dtype, device=device | torch.empty(out_features, dtype=dtype, device=device
formulas/comments:
  Step 1: Compute SRAM-resident Z = X @ V^T of shape [BLOCK_M, R]
  Step 3: Up-projection Y = Z_norm @ U^T + bias: [BLOCK_M, R] @ [R, BLOCK_K_OUT]
features: gating,gqa,lowrank,rank,rope,stride,svd
loc: 459