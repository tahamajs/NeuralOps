# d174 gfws  [compressor]
aliases: gfws,gfws_d
file: Codes/framework/compression_engine/gfws_d_compressor.py  doc: NONE
doc: GFWSVD: Fisher-weighted / Covariance-weighted SVD. Registered as 'gfws' and 'gfws_d'.
class GFWSLoRALinear(Module)
  doc: Linear layer containing Fisher/Covariance-whitened SVD compressed base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class GFWSCompressor(BaseCompressor)
  doc: Generalized Fisher / Covariance-Weighted SVD Compressor. | Whitens weights using empirical input activation covariance square-root Cov(X)^{1/2},
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, rank, damping, max_error, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, target_attention_only, tokenizer, calibration_loader, calibration_texts)
tensor_ops:
  x_flat = x.reshape(-1, self.in_features)
  return out.reshape(*orig_shape[:-1], self.out_features)
  flat_x = x.reshape(-1, in_f)
  batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
  M = torch.matmul(W, cov_sqrt)
  candidate = torch.matmul(M_r, cov_inv_sqrt)
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
formulas/comments:
  Whiten weights: M = W * Cov(X)^{1/2}
  De-whiten: candidate = M_r * Cov(X)^{-1/2}
features: calib,fisher,gqa,lora,peft,rank,svd
loc: 516