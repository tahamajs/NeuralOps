# d104 ptp  [attention]
aliases: progressive_token_pruning,ptp
file: Codes/framework/attentions/progressive_token_pruning.py  doc: Docs/content/attentions/progressive_token_pruning.md
doc: Progressive Token Pruning (PTP) – Production-Hardened. | Learnable Progressive Token Pruning module with guaranteed [CLS] retention, | multi-head attention capabilities, and full multi-architecture support. | Supports:
class LoRALinear(Module)
class ProgressiveTokenPruning(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, d_model, keep_ratio, rank, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  scores = scores.masked_fill(~valid_mask, -1e4)
  q = self.q_lin(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k_all = self.k_lin(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v_all = self.v_lin(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
  attn_scores = torch.matmul(q.float(), k_pruned.float().transpose(-2, -1)) * scale  # (B, H, L, k_tokens)
  q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
  attn_scores = attn_scores.masked_fill(causal_mask, -1e4)
shapes: [B, L
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation,topk
loc: 525