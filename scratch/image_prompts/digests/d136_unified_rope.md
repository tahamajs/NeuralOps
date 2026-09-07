# d136 unified_rope  [attention]
aliases: compression_unified_rope,unified_rope,unified_rope_attention
file: Codes/framework/attentions/unified_rope.py  doc: Docs/content/attentions/unified_rope.md
doc: Universal Rotary Position Embedding Attention (UnifiedRoPE). | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class UnifiedRoPE(BaseAttention)
  doc: Universal Rotary Position Embedding Attention module.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, dim, num_heads, max_seq_len, base, dropout, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  freqs = torch.einsum("i,j->ij", t, self.inv_freq)
  q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
  attn_weights = F.softmax(scores, dim=-1)
  head_mask = head_mask.view(1, -1, 1, 1)
  head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rope_partial,rotation
loc: 635