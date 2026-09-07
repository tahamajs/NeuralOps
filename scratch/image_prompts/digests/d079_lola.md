# d079 lola  [attention]
aliases: lola,lola_attention,lola_attn
file: Codes/framework/attentions/lola_attention.py  doc: Docs/content/attentions/lola_attention.md
doc: LoLA: Low-Rank Linear Attention with Sparse Caching. | Combines factorized low-rank projections with selective multi-system cached retrieval, | trainable LoRA residual adapters, frozen SVD base weights, GQA, RoPE, and causal decoding.
class LowRankCache(Module)
class LowRankLinear(Module)
class LoLAAttention(BaseAttention)
__init__(max_len, low_rank_k)
__init__(in_features, out_features, rank, lora_rank, bias)
forward(x)
__init__(config, model_config, rank, lora_rank, dropout, comp_config, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  y = torch.matmul(rk, q_proj).float()
  self.low_rank_k_buf = torch.matmul(q_orth, torch.matmul(q_orth.T, rk)).to(self.recent_k.dtype)
  y_v = torch.matmul(rv, q_proj).float()
  self.low_rank_v_buf = torch.matmul(q_orth_v, torch.matmul(q_orth_v.T, rv)).to(self.recent_v.dtype)
  w = torch.matmul(self.U, self.V)
  w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
  Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
shapes: [w, b
parameters: torch.empty(out_features, rank | torch.empty(rank, in_features | torch.empty(out_features | torch.empty(self.lora_rank, in_features | torch.empty(out_features, self.lora_rank
features: causal,distill,gating,gqa,lora,lowrank,peft,rank,rope,rotation,sparsity,ssm,svd
loc: 584