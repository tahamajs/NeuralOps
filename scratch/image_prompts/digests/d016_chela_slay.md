# d016 chela_slay  [attention]
aliases: chela_slay,chela_slay_attention
file: Codes/framework/attentions/chela_slay.py  doc: Docs/content/attentions/chela_slay.md
doc: CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel. | Implements hybrid linear attention via spherical normalization onto the unit hypersphere | combined with trigonometric random Fourier feature mappings (Yat-Kernel) and | adaptive gating for smooth distillation from pretrained attention.
class LoRALinear(Module)
  doc: Linear projection layer equipped with frozen base weights and trainable residual LoRA adapters. | Base path:
class ChelaSlayAttention(BaseAttention)
  doc: CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel. | Mathematical Formulation:
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, feature_dim, pretrained_attn, old_attn, gate_init_bias, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class DummyConfig()
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  proj = torch.matmul(x, self.random_proj.to(device=x.device, dtype=x.dtype))
  q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.W_k(kv_input).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.W_v(kv_input).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
  dense_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.head_dim**-0.5)
shapes: [B, H, L, 1 [B, L, H
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: adaptive,causal,distill,gating,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 670