# d131 svd_proj_attn  [attention]
aliases: svd_proj_attn,svd_projection,svd_projection_attention
file: Codes/framework/attentions/svd_projection_attention.py  doc: Docs/content/attentions/svd_projection_attention.md
doc: SVD Projection Attention. | Each projection (Q, K, V, Out) is factored into low-rank SVD components U and V, | with trainable LoRA adapters for recovery fine-tuning. | Hardened for production use across diverse transformer architectures:
class SVDProjectionLinear(Module)
  doc: Low-Rank SVD Linear Layer with Trainable LoRA Residual Adapters. | Forward computes:
class SVDProjectionAttention(BaseAttention)
  doc: Multi-Head Attention using Factorized SVD Linear Projections. | Hardened for production use across diverse transformer architectures.
__init__(in_features, out_features, rank, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, old_attn, rank, lora_rank, lora_alpha, is_decoder, dropout)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  return torch.matmul(self.U, self.V)
  w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
parameters: torch.empty(out_features, rank | torch.empty(rank, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,distill,gqa,lora,peft,pool,rank,rope,rope_partial,rotation,svd
loc: 661