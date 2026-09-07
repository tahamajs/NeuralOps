# d095 nsa  [attention]
aliases: nsa,nsa_attention
file: Codes/framework/attentions/nsa_attention.py  doc: Docs/content/attentions/nsa_attention.md
doc: Native Sparse Attention (NSA) Module (Production-Hardened). | Combines sliding-window local context capture with dynamic, input-dependent top-$k$ | global routing across key-value representations in a fully vectorized manner. | Supports:
class LoRALinear(Module)
class NativeSparseAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, window_size, top_k, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
  K = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3)
  V = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale  # [B, H, L, L]
  non_local_scores = scores.masked_fill(local_mask_4d | (~key_valid_4d), -1e4)
  non_local_scores = non_local_scores.masked_fill(~causal_tril, -1e4)
  scores = scores.masked_fill(~total_mask, -1e4)
shapes: [B, H, L, L [B, L, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation,sparsity,topk,window
loc: 517