# d051 inject_regla  [attention]
aliases: inject_regla
file: Codes/framework/attentions/regla_attention.py  doc: NONE
doc: Refined Gated Linear Attention (ReGLA) – Production-Hardened. | Combines element-wise key gating, normalized linear attention kernel evaluation, | and short-range depthwise 1D convolutions to capture both global associative memory and local inductive biases. | Supports:
class LoRALinear(Module)
class ReGLAAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, kernel_size, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
  k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  g = self.gate_proj(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  kv = torch.matmul(k_gated.transpose(-2, -1), v)
  num = torch.matmul(q, kv)
  den = torch.matmul(q, z.transpose(-2, -1)).clamp(min=1e-5)
  out = out * head_mask.view(1, self.num_heads, 1, 1).to(out.dtype)
shapes: [B, 1, L, 1 [B, D, L [B, H, L, d [B, H, L, d, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation
loc: 541