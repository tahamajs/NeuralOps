# d041 halo_regla  [attention]
aliases: halo_regla,halo_regla_attention
file: Codes/framework/attentions/halo_regla.py  doc: Docs/content/attentions/halo_regla.md
doc: HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention – Production-Hardened. | Combines multi-head linear projection with Rotary Position Embeddings (RoPE), | rectified kernel representations, and adaptive gating for zero-shot distillation from pretrained transformers. | Supports:
class LoRALinear(Module)
class HALOReGLAAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, hybrid_ratio, pretrained_attn, gate_init_bias)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  dense_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  dense_attn = F.softmax(dense_scores, dim=-1).to(hidden_states.dtype)
  dense_out = torch.matmul(dense_attn, v)  # (B, H, L, d)
  kv = torch.matmul(k_lin.transpose(-2, -1), v_masked)
  num = torch.matmul(q_lin, kv)
shapes: [B, 1, L, 1 [B, H, L, 1 [B, H, L, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: adaptive,causal,distill,gating,gqa,lora,peft,rank,rope,rotation
loc: 613