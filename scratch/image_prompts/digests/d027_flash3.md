# d027 flash3  [attention]
aliases: flash3,flash_attention3
file: Codes/framework/attentions/flash_attention3.py  doc: Docs/content/attentions/flash_attention3.md
doc: FlashAttention-3: Fast and Memory-Efficient Exact Attention with Asynchronous Hardware Overlap. | Integrates FlashAttention-3 kernels (with Hopper TMA asynchronous memory copy and FP8 precision support) | alongside graceful fallback to PyTorch's hardware-accelerated Scaled Dot-Product Attention (SDPA) and exact pretrained weight copying. | Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class FlashAttention3(BaseAttention)
  doc: FlashAttention-3 Multi-Head Exact Attention.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, use_fp8, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, Dh).transpose(1, 2)
  v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, Dh).transpose(1, 2)
  q_fa = q.transpose(1, 2).contiguous()
  k_fa = k.transpose(1, 2).contiguous()
shapes: [B, L, H, Dh
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 580