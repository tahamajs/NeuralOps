# d216 tomoe  [compressor]
aliases: tomoe,tomoe_compressor
file: Codes/framework/compression_engine/tomoe_compressor.py  doc: Docs/content/compressions/tomoe_compressor.md
doc: ToMoE: converts dense linear and Conv1D layers (e.g. FFN and attention projections) | into a Mixture of Experts using weight clustering. | Registered as 'tomoe' and 'tomoe_compressor'. | Production-hardened with:
class ToMoELinear(Module)
  doc: A single MoE-fied linear layer: learned routing + expert sub-layers. | Features:
class ToMoECompressor(BaseCompressor)
  doc: ToMoE Compressor: dynamically replaces dense feed-forward and projection layers | with an MoE-fied equivalent.
__init__(original, in_features, out_features, weight, bias, num_experts, top_k, lora_r, lora_alpha, freeze_experts, max_error)
forward(x)
__init__(model, num_experts, top_k, max_error, lora_r, lora_alpha, freeze_experts, target_attention_only, target_modules)
tensor_ops:
  x_flat = x.reshape(-1, in_feat)
  topk_weights = torch.softmax(topk_vals, dim=-1).to(x.dtype)  # (N, top_k)
  expanded_idx = topk_idx.unsqueeze(-1).expand(N, self.top_k, D_exp)
  index=dest_cols.reshape(N, self.top_k * D_exp),
  src=weighted_selected.reshape(N, self.top_k * D_exp),
  return out.reshape(*orig_shape[:-1], self.out_features)
  parts = full_name.split(".")
parameters: torch.empty(self.lora_r, self.in_features, device=device, dt | torch.zeros(n_neurons, self.lora_r, device=device, dtype=dty | torch.empty(self.lora_r, self.in_features, device=device, dt | torch.zeros(self.num_experts, self.lora_r, device=device, dt
formulas/comments:
  Conv1D stores weight as (nx, nf) where nx=in_features, nf=out_features
features: chunked,cluster,gating,gqa,lora,moe,peft,pool,rank,rope,topk
loc: 617