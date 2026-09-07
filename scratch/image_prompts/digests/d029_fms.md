# d029 fms  [attention]
aliases: factorised_matrix_state,fms
file: Codes/framework/attentions/factorised_matrix_state.py  doc: Docs/content/attentions/factorised_matrix_state.md
doc: Factorised Matrix State Attention Module. | Implements multi-head bidirectional/causal associative matrix memory updates combined with | learnable decay gating, denominator normalization, low-rank factorized output projection, | and pretrained weight transfer.
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class FactorisedMatrixState(BaseAttention)
  doc: Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, num_heads, pretrained_attn, old_attn, comp_config, layer_id, layer_idx)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  d_val = decay_f.view(1, H, 1, 1)
  S = d_val * S + torch.matmul(k_t, v_t.transpose(-1, -2))
  num = torch.matmul(S.transpose(-1, -2), q_t)  # [B, H, d, 1]
  denom = torch.matmul(z.transpose(-1, -2), q_t) + 1e-5
  q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
shapes: [1, H, 1, 1 [B, 1, L [B, H, d, 1 [B, H, d, d [B, L, H, d [H, 1, 1
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.zeros(self.all_head_size, self.proj_rank
formulas/comments:
  Weight verification (Frobenius norm >= 1e-4)
  State update: S_t = decay * S_{t-1} + k_t @ v_t^T
  Readout: y_t = (S_t @ q_t) / (z_t^T @ q_t + 1e-5)
features: causal,decay,distill,gqa,lora,lowrank,peft,rank,rope,rope_partial,rotation
loc: 618