# d069 inject_vig  [attention]
aliases: inject_vig
file: Codes/framework/attentions/vig_attention.py  doc: NONE
doc: ViG Attention: Vision GNN-inspired 2D Grid Linear Attention with Gated Fusion. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class ViGAttention(BaseAttention)
  doc: Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
  k = self.k_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)
  v = self.v_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)
  gates = torch.sigmoid(self.gate_net(x)).view(B, N_pad, 2, Hh, 1)
  q = q.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
  k = k.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
  v = v.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
  q_row = q.reshape(B, Hh * W, H, Dh).transpose(2, 3)
shapes: [B * Hh * H, W [B * Hh * H, W, Dh [B, H, W [B, Hh * W, Dh, H [B, Hh * W, H [B, Hh * W, H, Dh [B, Hh, H, W [B, Hh, H, W, Dh
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation
loc: 713