# d142 zeros  [attention]
aliases: zeros,zeros_attention
file: Codes/framework/attentions/zeros_attention.py  doc: Docs/content/attentions/zeros_attention.md
doc: Zero-Mean Linear Attention (ZeroS Attention). | Removes the mean from key representations and applies normalized linear attention with ReLU-based non-negative | feature maps, with multi-architecture weight transfer, trainable LoRA residual adapters, GQA, RoPE, and causal decoding. | Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
class LoRALinear(Module)
class ZeroSAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, lora_rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.W_q(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  K = self.W_k(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2)
  V = self.W_v(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2)
  step = torch.arange(1, N_k + 1, device=device, dtype=K.dtype).view(1, 1, N_k, 1)
  kv = torch.einsum("b h j d, b h j e -> b h j d e", k_hat, V)  # [B, H, N_k, d, d]
  num = torch.einsum("b h l d, b h l d e -> b h l e", q_hat, kv_cum)  # [B, H, L, d]
  num = torch.einsum("b h l d, b h l d e -> b h l e", q_hat, kv_cum_slice)  # [B, H, L, d]
  sim = torch.matmul(q_hat, k_hat.transpose(-1, -2))  # [B, H, L, N_k]
shapes: [B, H, L, 1 [B, H, L, d [B, H, d, d [B, L, H, Dh [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Apply RoPE: Priority 1 = position_embeddings, Priority 2 = rotary_emb fallback
features: causal,distill,gqa,lora,rank,rope,rotation
loc: 610