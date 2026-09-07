# d113 salt  [attention]
aliases: salt,salt_attention
file: Codes/framework/attentions/salt_attention.py  doc: Docs/content/attentions/salt_attention.md
doc: SALT Attention: Singular Value Adaptation with Low-Rank Tuning. | Parameter-efficient adaptation mechanism for transformer attention projections combining | orthogonal singular vector preservation, singular value rescaling ($\gamma$), and additive low-rank adapters ($A B$). | Hardened for multi-architecture production deployment.
class SALTLinear(Module)
  doc: Linear projection layer with SVD singular value modulation and low-rank residual branch. | Formulation:
class SALTAttention(BaseAttention)
  doc: SALT Attention module replacing standard linear projections with SALTLinear. | Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
__init__(in_features, out_features, rank, original_weight, original_bias)
forward(x)
__init__(config, rank, pretrained_attn, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  self.register_buffer("U", U[:, :r].contiguous().to(device=target_dev, dtype=target_dtype))
  self.register_buffer("Vt", Vt[:r, :].contiguous().to(device=target_dev, dtype=target_dtype))
  self.register_buffer("S", S[:r].contiguous().to(device=target_dev, dtype=target_dtype))
  final_w = w_oriented.view(out_dim, rep, -1).mean(dim=1)
  final_b = b.view(out_dim, rep).mean(dim=1) if b is not None else None
  Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, L_kv, self.num_kv_heads, Dh).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, L_kv, self.num_kv_heads, Dh).transpose(1, 2)
shapes: [B, L, H, Dh [w, b
parameters: torch.ones(r, device=target_dev, dtype=target_dtype | torch.randn(out_features, r, device=target_dev, dtype=target | torch.zeros(r, in_features, device=target_dev, dtype=target_ | original_bias.clone( | torch.zeros(out_features, device=target_dev, dtype=target_dt
features: causal,distill,gqa,lora,rank,rope,rope_partial,rotation,svd
loc: 540