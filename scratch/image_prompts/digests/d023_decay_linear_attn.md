# d023 decay_linear_attn  [attention]
aliases: decay_linear_attention,decay_linear_attn
file: Codes/framework/attentions/decay_linear_attention.py  doc: Docs/content/attentions/decay_linear_attention.md
doc: Decay Linear Attention (DecayLinearAttention). | Multi-head bidirectional & causal linear recurrent attention with data-dependent exponential decay gating, | ELU+1 feature maps, denominator normalization, and pretrained weight transfer. | Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class DecayLinearAttention(BaseAttention)
  doc: Decay Linear Attention Module with Data-Dependent Exponential Decay Gating.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, num_heads, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  S = lam_t * S + torch.matmul(k_t, v_t.transpose(-1, -2))
  num = torch.matmul(S.transpose(-1, -2), q_t)
  denom = torch.matmul(z.transpose(-1, -2), q_t) + 1e-5
  hidden_states = hidden_states.contiguous()
  q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
shapes: [B, H, 1, 1 [B, H, L [B, H, d, 1 [B, H, d, d [B, L, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Frobenius norm verification (>= 1e-4)
features: causal,decay,distill,gating,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 618