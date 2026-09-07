# d083 mamba_svd_flash  [attention]
aliases: mamba_svd_flash
file: Codes/framework/attentions/mamba_svd_flash.py  doc: Docs/content/attentions/mamba_svd_flash.md
doc: MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention Module (Production-Hardened). | Fuses bidirectional selective state-space recurrent dynamics with online Singular Value Decomposition | of state transitions, coupled with a Multi-Head / GQA FlashAttention execution path, | pretrained weight transfer, LoRA adapters, RoPE rotary embeddings, and causal/bidirectional masking.
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class MambaSVDFlash(BaseAttention)
  doc: MambaSVD-Flash Hybrid Bidirectional Attention Module. | Fuses bidirectional selective state-space recurrent dynamics with online Singular Value Decomposition of
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, hidden_size, num_heads, comp_config, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  v_t_1 = torch.bmm(V_t, h.unsqueeze(-1)).squeeze(-1)  # [B, r]
  h_next = torch.bmm(U_t, s_t_1.unsqueeze(-1)).squeeze(-1) + b_t * x_t
  m_t = mask[:, t].view(b, 1)
  hidden_states = hidden_states.contiguous()
  U = self.proj_u(hidden_states).view(B, L, D, self.rank)
  V = self.proj_v(hidden_states).view(B, L, self.rank, D)
  Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
shapes: [B, D [b, 1
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
formulas/comments:
  v_{t-1} = V_{t, r}^\top h_{t-1}
features: causal,distill,gqa,lora,peft,rank,rope,rotation,ssm,svd
loc: 757