# d144 aasvd  [compressor]
aliases: aa_svd,aa_svd_compressor,aasvd
file: Codes/framework/compression_engine/aa_svd_compressor.py  doc: Docs/content/compressions/aa_svd_compressor.md
doc: Activation-Aware Singular Value Decomposition (AASVD) Compressor. | Implements streaming Exponential Moving Average (EMA) activation covariance accumulation: | $$C_X^{(t)} = \beta C_X^{(t-1)} + (1 - \beta) \frac{1}{B} \sum_{i=1}^B X_i^\top X_i$$ | Projects weights onto principal activation subspaces and replaces dense `nn.Linear` layers
class SVDLinear(Module)
  doc: FIX: True Low-Rank Factorized Linear Layer executing as: | y = (x @ V.T) @ U.T + b
class AASVDCompressor(BaseCompressor)
  doc: Activation-Aware SVD Compressor with streaming EMA covariance accumulation | and true structural module replacement (SVDLinear).
__init__(in_features, out_features, rank, bias, use_fused_kernel, dtype, device, lora_r, lora_alpha)
forward(x)
__init__(model, tokenizer, rank, ema_beta, max_error, fallback_to_original, calibration_texts, device, target_modules, target_attention_only)
tensor_ops:
  "    scores = torch.matmul(q, k.transpose(-1, -2)) * scale\n"
  "    probs = torch.softmax(scores, dim=-1)\n"
  "    return torch.matmul(probs, v)\n"
  return torch.matmul(self.U, self.V)
  W_weighted = torch.matmul(W, cov_sqrt)
  V_r = torch.matmul(V_r, cov_inv_sqrt)
  W_recon = torch.matmul(U_r, V_r)
  x_flat = x_fp32.reshape(-1, in_f)
parameters: torch.empty(rank, in_features, dtype=dtype, device=device | torch.empty(out_features, rank, dtype=dtype, device=device | torch.empty(out_features, dtype=dtype, device=device | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: calib,gating,lora,lowrank,peft,pool,rank,rope,svd
loc: 627