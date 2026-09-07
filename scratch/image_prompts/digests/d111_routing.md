# d111 routing  [attention]
aliases: routing,routing_attention
file: Codes/framework/attentions/routing_attention.py  doc: Docs/content/attentions/routing_attention.md
doc: Routing Attention: Content-Based Clustering for Sparse Attention – Production-Hardened. | Partitions sequence queries and keys into discrete clusters via online k-means, | confining quadratic attention computations to intra-cluster token neighborhoods with exact masking. | Supports:
class LoRALinear(Module)
class RoutingAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, num_clusters, n_iters, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
  Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
  K = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3)
  V = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(~total_mask, -1e4)
  attn = F.softmax(scores, dim=-1).to(dtype)
  attn = attn * head_mask.view(1, self.num_heads, 1, 1).to(attn.dtype)
shapes: [B, L, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,cluster,distill,gqa,lora,peft,rank,rope,rotation,sparsity
loc: 528