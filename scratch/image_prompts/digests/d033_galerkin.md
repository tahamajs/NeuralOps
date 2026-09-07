# d033 galerkin  [attention]
aliases: galerkin,galerkin_attention
file: Codes/framework/attentions/galerkin_attention.py  doc: Docs/content/attentions/galerkin_attention.md
doc: Galerkin Attention: Petrov-Galerkin Projection Attention for Transformer Architectures. | Formulates transformer self-attention as a continuous non-local integral operator | discretized via Petrov-Galerkin projection onto an orthonormal basis (e.g., discrete cosine basis). | Supports fallback to standard causal attention for decoder models.
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class GalerkinAttention(BaseAttention)
  doc: Petrov-Galerkin Projection Attention for Transformer Architectures.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, dropout, basis_type, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class DummyConfig()
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
  K = self.key(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2)
  V = self.value(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale  # [B, H, N, N]
  probs = F.softmax(scores, dim=-1)
shapes: [B, H, N, N [B, H, N, d [B, N, H, d [N, N
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Weight verification (Frobenius norm >= 1e-4)
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 606