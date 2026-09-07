# d149 armor  [compressor]
aliases: armor,armor_compressor,armor_pruner
file: Codes/framework/compression_engine/armor_pruner.py  doc: Docs/content/compressions/armor_pruner.md
doc: ARMOR: Adaptive Resilient Matrix Orthogonal Rank Pruning (2:4 Sparse Core + Low-Rank Correction). | Registered in COMPRESSOR_REGISTRY as 'armor_pruner', 'armor', and 'armor_compressor'. | Mathematical Formulation: | 1. Resilient First-Order Taylor Saliency (handling zero-grad, NaN/Inf, and absent activations):
class ARMORLinear(Module)
  doc: Decoupled Sparse + Low-Rank Residual Layer with optional LoRA scaling. | Maintains clean gradient flow to low-rank adapter branches while base sparse weights are frozen.
class ARMORPruner(BaseCompressor)
  doc: Production-Hardened ARMOR Pruner: Combines 2:4 structured sparsity with low-rank SVD error compensation | and decoupled dual-branch gradient propagation.
__init__(in_features, out_features, rank, bias, lora_alpha, dtype, device)
forward(x)
__init__(model, block_size, rank, sparsity, structured_24, decouple, recalibrate_ln, lora_alpha, max_error, fallback_to_original, target_attention_only, skip_attention_qkv)
tensor_ops:
  X = activations.reshape(-1, activations.shape[-1]).float()
  return self.sparse_weight * self.sparse_mask + self.scaling * torch.matmul(self.lora_B, self.lora_A)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  aggregated[name] = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0)
  flat = saliency.reshape(-1)
  correction = torch.matmul(lora_B, lora_A)
  W_recon = W_sparse + scaling * torch.matmul(lora_B.float(), lora_A.float())
parameters: torch.empty(out_features, in_features, dtype=dtype, device=d | torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.empty(out_features, dtype=dtype, device=device
features: adaptive,calib,gating,gqa,lora,lowrank,peft,pool,rank,rope,sparsity,svd,topk
loc: 671