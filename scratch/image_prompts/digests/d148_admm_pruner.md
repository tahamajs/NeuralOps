# d148 admm_pruner  [compressor]
aliases: admm_pruner,admm_pruner_distill
file: Codes/framework/compression_engine/admm_pruner.py  doc: Docs/content/compressions/admm_pruner.md
doc: ADMM Pruner: Alternating Direction Method of Multipliers for Structured & Semi-Structured Pruning | with Layer-Wise Reconstruction Distillation, STE, and Dynamic Rho. | Registered in COMPRESSOR_REGISTRY as 'admm_pruner' and 'admm_pruner_distill'. | Mathematical Formulation:
class ADMMSTEPrune(Function)
  doc: Straight-Through Estimator (STE) autograd function for ADMM sparse weights. | Forwards masked weights and passes gradients through uninhibited on backward pass.
class ADMMMaskedLinear(Module)
  doc: Linear layer maintaining continuous weights and an ADMM-derived binary mask with STE gradient backprop.
class ADMMPruner(BaseCompressor)
  doc: Production-Hardened ADMM Pruner with Layer-Wise Reconstruction Distillation, | 2:4 semi-structured tensor core acceleration, STE gradients, and dynamic rho scheduling.
forward(weight, mask)
__init__(in_features, out_features, bias, dtype, device)
forward(x)
__init__(model, sparsity, rho, max_iter, structured_24, block_size, dynamic_rho, use_ste, recalibrate_ln, max_error, fallback_to_original, protect_critical)
tensor_ops:
  x_flat = x.reshape(-1, x.shape[-1]).float().cpu()
  flat = tensor.abs().reshape(-1)
  X_calib = X_calib.reshape(-1, in_features)
  res_w = W_final.t().contiguous().to(dtype=dtype, device=device)
  res_w = W_final.contiguous().to(dtype=dtype, device=device)
parameters: torch.empty(out_features, in_features, dtype=dtype, device=d | torch.empty(out_features, dtype=dtype, device=device
formulas/comments:
  Pre-compute target uncompressed outputs: Y0 = X @ W0^T
features: adaptive,calib,distill,gating,hash,lora,peft,pool,sparsity,topk
loc: 762