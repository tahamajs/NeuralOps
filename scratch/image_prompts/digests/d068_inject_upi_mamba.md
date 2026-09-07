# d068 inject_upi_mamba  [attention]
aliases: inject_upi_mamba
file: Codes/framework/attentions/upi_stuffed_mamba.py  doc: NONE
doc: UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM Block. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class UPIMambaBlock(BaseAttention)
  doc: Bidirectional / Causal Uncertainty-Penalized Interpolated Mamba Block.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, dim, d_state, d_conv, expand, lora_rank, lora_alpha, is_decoder, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  m_t = mask[:, t].view(B, 1, 1).float()
  x_in, z = xz.chunk(2, dim=-1)
  x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
  dt_raw, gamma_raw = x_dbl.chunk(2, dim=-1)
shapes: [B, 1, 1
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.log(torch.arange(1, d_state + 1, dtype=torch.float32
features: causal,chunked,distill,gating,gqa,lora,rank,rope,ssm
loc: 533