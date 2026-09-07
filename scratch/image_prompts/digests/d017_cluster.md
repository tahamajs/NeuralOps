# d017 cluster  [attention]
aliases: cluster,cluster_attention
file: Codes/framework/attentions/cluster_attention.py  doc: Docs/content/attentions/cluster_attention.md
doc: ClusterFormer: Token Clustering Attention Module. | Groups queries into $K$ semantic cluster centroids via fast differentiable $K$-means | and evaluates centroid-to-key cross-attention alongside local context, achieving $\mathcal{O}(K \cdot L \cdot d)$ linear complexity | while eliminating token isolation. For causal decoders, strict causal masking is enforced.
class LoRALinear(Module)
class ClusterAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, num_clusters, n_iters, window_size, lora_rank, comp_config, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
  Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2)
  pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
shapes: [B, 1, 1, L [B, H, K, Dh [B, H, K, L [B, H, K, d [B, H, L [B, H, L, Dh [B, H, L, d [B, L, H, Dh
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor(0.5
features: causal,cluster,distill,gqa,lora,peft,rank,rope,rotation,window
loc: 705