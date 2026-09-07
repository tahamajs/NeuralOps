# d034 gated_delta_net  [attention]
aliases: gated_delta_net
file: Codes/framework/attentions/gated_delta_net.py  doc: Docs/content/attentions/gated_delta_net.md
doc: Gated DeltaNet Attention Module. | Implements multi-head bidirectional/causal Delta Rule linear recurrent attention with | input-dependent exponential decay gating, associative state error correction, and | pretrained self-attention weight transfer.
class LoRALinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapters.
class GatedDeltaNet(BaseAttention)
  doc: Bidirectional & Causal Gated DeltaNet Recurrent Linear Attention with True O(L) Complexity.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, num_heads, comp_config, layer_id, layer_idx, old_attn, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
  w = w.reshape(w.size(0), -1)
  pred_v = torch.matmul(S_fwd, k_t)
  S_fwd = b_t * S_fwd + a_t * torch.matmul(e_t, k_t.transpose(-1, -2))
  y_t = torch.matmul(S_fwd, q_t).squeeze(-1)
  pred_v = torch.matmul(S_bwd, k_t)
  S_bwd = b_t * S_bwd + a_t * torch.matmul(e_t, k_t.transpose(-1, -2))
shapes: [B, 1, L [B, H, 1, 1 [B, H, L [B, H, L, d [B, H, d, 1 [B, H, d, d [B, L, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  Weight verification (Frobenius norm >= 1e-4)
  1. Forward recurrent scan: S_t = beta_t * S_{t-1} + alpha_t * (v_t - S_{t-1} k_t) k_t^T
  Error correction: e_t = v_t - S_{t-1} @ k_t
  Readout: y_t = S_t @ q_t
features: causal,decay,delta,distill,gating,gqa,lora,peft,rank,rope,rope_partial,rotation
loc: 771