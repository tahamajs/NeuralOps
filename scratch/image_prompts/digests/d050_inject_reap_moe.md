# d050 inject_reap_moe  [attention]
aliases: inject_reap_moe
file: Codes/framework/attentions/reap_moe.py  doc: NONE
doc: REAP-MoE: Robust Expert Allocation and Pruning Mixture of Experts Attention Module – Production-Hardened. | Implements a vectorized Mixture-of-Experts block with load-balancing regularization, | SVD low-rank expert factorization, and multi-head attention weight transfer. | Supports:
class LoRALinear(Module)
class SVDLinear(Module)
class REAPMoE(Module)
class REAPMoEAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(in_features, out_features, rank)
forward(x)
__init__(d_model, n_experts, top_k, noise_std, use_svd, config)
forward(x)
__init__(config, rank, n_experts, top_k, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  topk_weights = F.softmax(topk_weights, dim=-1)
  router_probs = F.softmax(logits / 1.0, dim=-1).view(-1, self.n_experts)
  density = expert_mask.sum(dim=2).view(-1, self.n_experts).mean(dim=0)
  topk_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
  q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.k_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.v_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
shapes: [B, L, E, D [B, L, k, D
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.randn(out_features, self.rank
features: causal,distill,gqa,lora,moe,peft,rank,rope,rotation,svd,topk
loc: 624