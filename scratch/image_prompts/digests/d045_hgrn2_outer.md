# d045 hgrn2_outer  [attention]
aliases: hgrn2_outer
file: Codes/framework/attentions/hgrn2_outer.py  doc: Docs/content/attentions/hgrn2_outer.md
doc: HGRN-2 Outer-Product Recurrent Linear Attention Module. | Implements multi-head bidirectional/causal outer-product associative matrix memory updates | with log-space bounded decay gating, denominator normalization, and pretrained weight transfer. | Hardened for production use across diverse transformer architectures:
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class HGRN2OuterBlock(BaseAttention)
  doc: HGRN-2 Outer-Product Recurrent Linear Attention Module. | Maintains multi-head outer-product associative matrix recurrence with input-dependent decay gating.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, dim, num_heads, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  attn_scores = q_f @ k_f.transpose(-1, -2)   # [B, H, L, L]
  hidden_states = hidden_states.contiguous()
  q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)
  f = torch.exp(-F.softplus(self.W_f(hidden_states))).transpose(1, 2)  # [B, H, L]
  head_mask = head_mask.view(1, -1, 1, 1)
  head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
shapes: [B, H, L [B, H, L, 1 [B, H, L, L [B, H, L, d [B, L, H * d [B, L, H, d
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,decay,distill,gqa,lora,rank,rope,rope_partial,rotation
loc: 550