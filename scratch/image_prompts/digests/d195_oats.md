# d195 oats  [compressor]
aliases: oats,oats_compressor
file: Codes/framework/compression_engine/oats_compressor.py  doc: Docs/content/compressions/oats_compressor.md
doc: OATS: Outlier-Aware Training & Saliency-Based Sparsification with Low-Rank Matrix Decomposition. | Registered in COMPRESSOR_REGISTRY as 'oats' and 'oats_compressor'. | Mathematical Formulation: | 1. Activation Second Moment:
class OATSLoRALinear(Module)
  doc: Linear layer containing OATS decomposed base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class OATSCompressor(BaseCompressor)
  doc: OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition. | Production hardening features:
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, tokenizer, sparsity, rank_ratio, num_iters, device, calibration_loader, calibration_texts, structured_24, max_error, fallback_to_original, skip_attention_qkv)
tensor_ops:
  X = activations.reshape(-1, activations.shape[-1]).float().to(device)
  M = torch.matmul(X.T, X) / max(X.shape[0], 1)
  W_scaled = torch.matmul(W, L_M)
  L = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
  flat = R.abs().reshape(-1)
  L = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
  flat = R.abs().reshape(-1)
  W_comp = torch.matmul(L + S_mat, L_M_inv)
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
formulas/comments:
  Scale weights: W_scaled = W @ L_M
  De-scale: W_comp = (L + S_mat) @ L_M^-1
features: calib,gating,gqa,lora,peft,rank,sparsity,svd,topk
loc: 570