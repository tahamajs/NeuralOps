# d106 rala  [attention]
aliases: rala,rala_attention,rala_linear_attention
file: Codes/framework/attentions/rala_attention.py  doc: Docs/content/attentions/rala_attention.md
doc: Rank‑Augmented Linear Attention (RALA) – Production-Hardened. | Augments linear attention kernel maps with low-rank static head interaction matrices. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
class RALALinearAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2)) * causal_mask
  intra_out = torch.matmul(intra_attn, v_c)
  inter_out = torch.matmul(q_c, S)
  S = S + torch.matmul(k_c.transpose(-1, -2), v_c)
  rank_mat = torch.einsum("h d r, h e r -> h d e", self.U, self.V)
shapes: [B, 1, N, 1 [B, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.randn(self.num_heads, self.head_dim, self.rank_r
formulas/comments:
  True linear-time recurrence via chunked scan (O(chunk_size * d^2) memory, no O(N * d^2) allocation)
features: causal,chunked,distill,gqa,lora,peft,rank,rope,rotation
loc: 569