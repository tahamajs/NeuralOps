# d173 frandi  [compressor]
aliases: frandi,frandi_compressor
file: Codes/framework/compression_engine/frandi_compressor.py  doc: Docs/content/compressions/frandi_compressor.md
doc: FRanDI: Feature-Regression Aligned Singular Value Decomposition Compressor. | Registered in COMPRESSOR_REGISTRY as 'frandi' and 'frandi_compressor'.
class FRanDILinear(Module)
  doc: Feature-Regression Aligned Factorized Linear Layer with optional LoRA adapter. | W ≈ B @ A
class FRanDICompressor(BaseCompressor)
  doc: FRanDI (Feature-Regression aligned SVD) Compressor. | Compresses linear layers using singular value decomposition with optional
__init__(in_features, out_features, rank, bias, lora_r, lora_alpha, dtype, device)
forward(x)
__init__(model, rank, energy_threshold, calibration_loader, lora_r, lora_alpha, max_error, target_attention_only, fallback_to_original, skip_attention_qkv, cfg, target_layers)
tensor_ops:
  return torch.matmul(self.B, self.A)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  return torch.matmul(self.B, self.A)
  W_tilde = torch.matmul(W_float, L)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  x_flat = x.reshape(-1, in_f)
  cov = torch.matmul(x_flat.t(), x_flat)
parameters: A_init.to(dtype=dtype | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.zeros(out_features, dtype=dtype, device=device | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: adaptive,calib,gating,gqa,lora,peft,pool,rank,rope,svd
loc: 546