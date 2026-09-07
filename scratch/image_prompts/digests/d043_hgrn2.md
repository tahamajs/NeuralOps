# d043 hgrn2  [attention]
aliases: hgrn2,hgrn2_layer
file: Codes/framework/attentions/hgrn2_layer.py  doc: Docs/content/attentions/hgrn2_layer.md
doc: Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer. | Implements bidirectional/causal element-wise linear recurrence with dynamic input-dependent | log-space forget gating, input modulation, LoRA adapters, and multi-architecture pretrained weight transfer.
class HGRN2Layer(BaseAttention)
  doc: Hierarchically Gated Recurrent Network v2 (HGRN-2) Layer. | Supports causal mode for decoders, bidirectional mode for encoders, LoRA adapters,
__init__(config, rank, d_model, res_clip, lora_rank, lora_alpha, is_decoder, freeze_base)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  chunks = in_w.chunk(3, dim=0)
  b_chunks = in_b.chunk(3, dim=0)
  C = torch.cumsum(log_f, dim=1).permute(0, 2, 1)            # [B, D, L]
  u = iv_fp32.permute(0, 2, 1).unsqueeze(-1)  # [B, D, L, 1]
  h_fwd = torch.matmul(D_causal, u).squeeze(-1).permute(0, 2, 1).to(iv.dtype)  # [B, L, D]
  h_bwd = torch.matmul(D_anti, u).squeeze(-1).permute(0, 2, 1).to(iv.dtype)    # [B, L, D]
  hidden_states = hidden_states.contiguous()
  v = v + torch.matmul(torch.matmul(hidden_states, self.v_lora_A.t()), self.v_lora_B.t()) * self.lora_scaling
shapes: [B, D, L [B, D, L, 1 [B, D, L, L [B, L, D
parameters: torch.randn(self.lora_rank, self.d_model | torch.zeros(self.d_model, self.lora_rank | torch.randn(self.lora_rank, self.d_model | torch.zeros(self.d_model, self.lora_rank
formulas/comments:
  Initialize forget gate bias to -3.0 so softplus(-3) ~ 0.048 => f_t ~ exp(-0.048) ~ 0.95
features: causal,chunked,decay,distill,gating,lora,peft,rank
loc: 468