# d135 tpa  [attention]
aliases: tp_attention,tpa,tpa_attention
file: Codes/framework/attentions/tpa_attention.py  doc: Docs/content/attentions/tpa_attention.md
doc: TPAttention: Tensor Product Attention with Factorised Query-Key Projections. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class TPAttention(BaseAttention)
  doc: Tensor Product Attention with factorised low-rank query-key projection.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q1 = self.q_proj1(hidden_states).view(B, L, H, r)
  q2 = self.q_proj2(hidden_states).view(B, L, H, r)
  k1 = self.k_proj1(hidden_states).view(B, L, H_kv, r)
  k2 = self.k_proj2(hidden_states).view(B, L, H_kv, r)
  v = self.v_proj(hidden_states).view(B, L, H_kv, hd)
  Q = torch.einsum("blhi,blhj->blhij", q1, q2).reshape(B, L, H, r * r).permute(0, 2, 1, 3)
  K = torch.einsum("blhi,blhj->blhij", k1, k2).reshape(B, L, H_kv, r * r).permute(0, 2, 1, 3)
  v = v.permute(0, 2, 1, 3)  # (B, H_kv, L, hd)
shapes: [B, L, H * hd
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
formulas/comments:
  Tensor product: Q = q1 ⊗ q2, shape [B, L, H, r*r]
features: causal,distill,gqa,lora,rank,rope,rope_partial,rotation,svd
loc: 619