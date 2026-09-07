# d021 cosformer  [attention]
aliases: cosformer,cosformer_attention
file: Codes/framework/attentions/cosformer.py  doc: Docs/content/attentions/cosformer.md
doc: Cosformer: Rethinking Softmax in Attention with Linear Complexity. | Implements non-negative kernel feature mapping with cosine/sine positional reweighting, | cumulative sum causal linear attention for decoders, and associative accumulation for encoders. | Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class CosformerAttention(BaseAttention)
  doc: Cosformer Attention with Linear Complexity and Exact Multi-Architecture Support.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, max_seq_len, act_fun, eps, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  cos_weight = torch.cos(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
  sin_weight = torch.sin(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
  Q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2)
shapes: [B, H, 1, d [B, H, N, d [B, H, N, d, d [B, H, d, d [B, N, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor(float(self.head_dim
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 641