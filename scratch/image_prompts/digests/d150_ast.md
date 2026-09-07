# d150 ast  [compressor]
aliases: ast,ast_pruner
file: Codes/framework/compression_engine/ast_pruner.py  doc: Docs/content/compressions/ast_pruner.md
doc: AST: Adaptive Semi-Structured 2:4 and N:M Pruning with Empirical Fisher Calibration and PCA Coordinate Rotation. | Registered in COMPRESSOR_REGISTRY as 'ast_pruner' and 'ast'. | Mathematical Formulation: | 1. Activation Covariance & PCA Coordinate Rotation:
class ASTPrunedLoRALinear(Module)
  doc: Linear layer containing AST 2:4 semi-structured pruned base weights (frozen) | with optional trainable LoRA adapters (lora_A, lora_B).
class ASTPruner(BaseCompressor)
  doc: AST Pruner: Vectorized semi-structured 2:4 and N:M pruning engine with | empirical Fisher calibration and PCA coordinate rotation.
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, block_size, n_keep, dim, use_pca, use_fisher, recalibrate_ln, max_error, fallback_to_original, protect_critical, lora_r, lora_alpha)
tensor_ops:
  x_flat = x.reshape(-1, self.in_features)
  return out.reshape(*orig_shape[:-1], self.out_features)
  act_concat = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0).float()
  cov = torch.matmul(centered.T, centered) / max(N - 1, 1)
  fisher_info[name] = torch.var(act_concat, dim=0, keepdim=True).expand(1, D)
  W_pca = torch.matmul(W, Q)
  F_pca = torch.matmul(F_mat, Q.pow(2))
  F_pca = torch.matmul(F_mat.reshape(1, d_in), Q.pow(2)).expand(d_out, d_in)
shapes: [1, D
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
formulas/comments:
  2. Rotate weights into PCA coordinate space: \widetilde{W} = W Q
  5. Rotate back to original coordinate basis: W = \widetilde{W}_pruned Q^T
features: adaptive,calib,fisher,gqa,lora,peft,pool,rotation,svd
loc: 552