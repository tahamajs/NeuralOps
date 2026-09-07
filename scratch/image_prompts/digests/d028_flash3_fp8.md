# d028 flash3_fp8  [attention]
aliases: flash3_fp8,flash_attention3_fp8
file: Codes/framework/attentions/flash_attention3_fp8.py  doc: Docs/content/attentions/flash_attention3_fp8.md
doc: FlashAttention-3 FP8: Low-Precision 8-bit Float Transformer Attention. | Implements FlashAttention-3 utilizing FP8 (e4m3fn) precision format for Hopper Tensor Cores, | with automatic fallback to hardware-accelerated SDPA. | Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class FlashAttention3FP8(BaseAttention)
  doc: FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, n_heads, use_fp8, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, Dh).transpose(1, 2)
  v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, Dh).transpose(1, 2)
  q_fa = q.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
  k_fa = k.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
shapes: [B, L, H, Dh
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 579