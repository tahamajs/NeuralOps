# d196 optimal_subspace_projection  [compressor]
aliases: optimal_subspace_projection
file: Codes/framework/compression_engine/optimal_subspace_projection.py  doc: Docs/content/compressions/optimal_subspace_projection.md
doc: Optimal Subspace Projection: Activation-Aware Subspace Projection via Randomized SVD. | Registered in COMPRESSOR_REGISTRY as 'optimal_subspace_projection'. | Mathematical Formulation: | 1. Activation Covariance:
class OptimalSubspaceLoRALinear(Module)
  doc: Linear layer containing optimal subspace projected base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class OptimalSubspaceProjectionCompressor(BaseCompressor)
  doc: Optimal Subspace Projection Compressor using activation-aware randomized SVD. | Production hardening features:
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, calibration_loader, rank_ratio, power_iters, tokenizer, calibration_texts, device, max_error, fallback_to_original, target_attention_only, target_modules, lora_r)
tensor_ops:
  x_flat = x.reshape(-1, x.shape[-1]).float()
  cov = torch.matmul(x_flat.T, x_flat) / max(x_flat.shape[0], 1)
  M = torch.matmul(W, L)
  Y = torch.matmul(M, Omega)
  Y = torch.matmul(M, torch.matmul(M.T, Y))
  B = torch.matmul(Q.T, M)
  new_U = torch.matmul(Q, U_r)
  new_V = torch.matmul(Vt_r, L_inv)
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: calib,gqa,lora,peft,rank,svd
loc: 505