# d001 adaptive_span  [attention]
aliases: adaptive_span,adaptive_span_attention
file: Codes/framework/attentions/adaptive_span_attention.py  doc: Docs/content/attentions/adaptive_span_attention.md
doc: Adaptive Span Attention Module (Sukhbaatar et al., 2019). | Mathematical Formulation: | 1. Query-Key Attention Scores: | $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$
class LoRALinear(Module)
class AdaptiveSpanAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, dropout, max_span, lora_rank, comp_config, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  values = values.permute([2, 0, 1]).unsqueeze(0)
  Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  pos_ids = torch.arange(N_q, device=device).unsqueeze(0).expand(B, -1)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
shapes: [H, N, N [N, N [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.full((self.num_heads,
features: adaptive,causal,distill,gqa,lora,rank,rope,rotation
loc: 679