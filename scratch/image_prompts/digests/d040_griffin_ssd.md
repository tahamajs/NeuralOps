# d040 griffin_ssd  [attention]
aliases: griffin_ssd,griffin_ssd_block
file: Codes/framework/attentions/griffin_ssd.py  doc: Docs/content/attentions/griffin_ssd.md
doc: Griffin SSD: Hybrid Local Sliding-Window Attention and Global State Space Duality (SSD) Kernel. | Combines bidirectional/causal local sliding-window self-attention with multi-head | State Space Duality linear recurrence, LoRA residual adapters, and multi-architecture pretrained weight transfer. | Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
class GriffinSSDBlock(BaseAttention)
  doc: Griffin Hybrid Block combining local window attention with global SSD linear attention.
__init__(config, rank, window_size, d_model, num_heads, old_attn, lora_rank, lora_alpha)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  attn_scores = (q @ k.transpose(-1, -2))  # [B, H, L, L]
  hidden_states = hidden_states.contiguous()
  q = self.W_q(hidden_states).view(B, L, H, d).transpose(1, 2)
  k = self.W_k(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)
  v = self.W_v(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)
  q_ssd = self.ssd_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
  scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)  # [B, H, L, L]
  scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), mask_val)
shapes: [1, H, 1, 1 [B, 1, 1, L [B, H, L, L [B, H, L, d [B, L, H, d [w, b
parameters: torch.ones(self.num_heads, 1, 1
features: causal,decay,distill,gating,gqa,lora,peft,rank,rope,rotation,window
loc: 549