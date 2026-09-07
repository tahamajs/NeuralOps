# d119 span_expanded  [attention]
aliases: span_expanded,span_expanded_attention
file: Codes/framework/attentions/span_expanded_attention.py  doc: Docs/content/attentions/span_expanded_attention.md
doc: Span-Expanded Attention with HyLoRA – Production-Hardened. | Extends attention receptive fields via fixed/cached key-value span expansion buffers, | complemented by parallel low-rank adaptation (HyLoRA) for parameter-efficient adaptation. | Supports:
class LoRALinear(Module)
class SpanExpandedAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, expansion_span, use_lora, lora_rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(B, L, H, Dh).permute(0, 2, 1, 3)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).permute(0, 2, 1, 3)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).permute(0, 2, 1, 3)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  attn = F.softmax(scores, dim=-1).to(hidden_states.dtype)
  attn = attn * head_mask.view(1, self.num_heads, 1, 1).to(attn.dtype)
  attn = attn * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(attn.dtype)
  context = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
shapes: [B, L, H, Dh
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 500