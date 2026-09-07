# d036 gated_linear_attn  [attention]
aliases: gated_linear_attention,gated_linear_attn
file: Codes/framework/attentions/gated_linear_attention.py  doc: Docs/content/attentions/gated_linear_attention.md
doc: Gated Linear Attention (GatedLinearAttention). | Multi-head linear recurrent attention with data-dependent decay gating, | multiplicative SiLU output gating, denominator normalization, and pretrained weight transfer. | Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
class LoRALinear(Module)
class GatedLinearAttention(BaseAttention)
  doc: Gated Linear Attention (GLA) Bidirectional & Causal Module. | Hardened for production across diverse transformer architectures.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, num_heads, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
  attn_scores = (q @ k.transpose(-1, -2)) * scale  # [B, H, L, L]
  hidden_states = hidden_states.contiguous()
  q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
  k = self.k_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2)
  v = self.v_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2)
  decay_logits = self.decay_gate(hidden_states).transpose(1, 2)  # [B, H, L]
  head_mask = head_mask.view(1, -1, 1, 1)
shapes: [B, H, L [B, H, L, 1 [B, H, L, L [B, H, L, d [B, L, H, d [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,decay,distill,gating,gqa,lora,rank,rope,rotation
loc: 576