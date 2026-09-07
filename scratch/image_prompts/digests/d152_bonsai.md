# d152 bonsai  [compressor]
aliases: bonsai,bonsai_pruner
file: Codes/framework/compression_engine/bonsai_pruner.py  doc: Docs/content/compressions/bonsai_pruner.md
doc: Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradient Descent Recovery & [CLS] Routing Protection. | Registered in COMPRESSOR_REGISTRY as 'bonsai' and 'bonsai_pruner'. | Production-hardened with: | - Multi-architecture support: DistilBERT, RoBERTa, GPT-2 (Conv1D), Qwen3.5 (GQA)
class BonsaiPrunedLoRALinear(Module)
  doc: Linear layer with sparse pruned base weights and optional trainable LoRA adapters. | Base weights are frozen with persistent sparse mask; LoRA adapters (lora_A, lora_B)
class BonsaiPruner(BaseCompressor)
  doc: Bonsai Pruner: Progressive multi-step iterative pruning with cubic schedule, | inter-step gradient descent recovery, GQA-aware protection, and [CLS] routing protection.
__init__(in_features, out_features, bias, lora_r, lora_alpha, freeze_base, dtype, device)
forward(x)
__init__(model, sparsity, steps, recovery_steps, recovery_lr, structured_24, block_size, protect_cls, skip_attention_qkv, target_attention_only, use_lora, lora_r)
tensor_ops:
  W_eff = W_eff + (torch.matmul(self.lora_B, self.lora_A) * self.scaling)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  last_part = name_lower.split(".")[-1]
  aggregated[name] = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0)
  W0 = raw_w.t().float().contiguous() if is_conv1d else raw_w.float().contiguous()
  flat = saliency.reshape(-1)
  cov_X = torch.matmul(X.T, X) / max(X.shape[0], 1)
parameters: torch.empty(out_features, in_features, dtype=dtype, device=d | torch.zeros(out_features, dtype=dtype, device=device | torch.empty(self.lora_r, in_features, dtype=dtype, device=de | torch.empty(out_features, self.lora_r, dtype=dtype, device=d
features: adaptive,calib,distill,gating,gqa,lora,peft,pool,rope,sparsity,topk
loc: 618