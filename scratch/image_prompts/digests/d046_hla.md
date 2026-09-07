# d046 hla  [attention]
aliases: hla
file: Codes/framework/attentions/hla_attention.py  doc: NONE
doc: Hadamard Linear Attention (HLA). | Multi-head bidirectional/causal linear recurrent attention with Hadamard associative kernelization, | learnable decay gating, denominator normalization, and pretrained weight transfer. | Hardened for production use across diverse transformer architectures:
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class HLAAttention(BaseAttention)
  doc: Hadamard Linear Attention (HLA) Bidirectional / Causal Module. | Maintains multi-head associative causal/anti-causal matrix states $S_t \in \mathbb{R}^{B \times H \times d_k \times d_k}$.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, d_model, num_heads, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  m_t = mask[:, t].view(B, 1, 1, 1).float()
  denom = (z.transpose(-1, -2) @ q_t.unsqueeze(-1)).squeeze(-1) + 1e-5
  hidden_states = hidden_states.contiguous()
  q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2)
  head_mask = head_mask.view(1, -1, 1, 1)
  head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
shapes: [1, H, 1, 1 [B, 1, 1, 1 [B, H, d [B, L, H * Dh [B, L, H, Dh
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank | torch.ones(self.num_heads, 1, 1
features: causal,decay,distill,gqa,hadamard,lora,rank,rope,rope_partial,rotation
loc: 536