# d114 samba_m3  [attention]
aliases: samba_m3
file: Codes/framework/attentions/samba_m3.py  doc: Docs/content/attentions/samba_m3.md
doc: SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Block. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class SpectralMamba(Module)
  doc: Mamba with learnable spectral gating.
class SAMBAM3Block(Module)
  doc: SAMBA-M3: Combination of Local Attention and Spectral Mamba.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(d_model, config)
forward(x)
__init__(config, d_model, n_heads, window, rank, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_lin(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
  k = self.k_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(causal_window_mask.unsqueeze(0).unsqueeze(0), -10000.0)
  scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), -10000.0)
  scores = scores.masked_fill(pad_mask, -10000.0)
  probs = F.softmax(scores, dim=-1)
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank | torch.zeros(d_model
features: causal,distill,gating,gqa,lora,rank,rope,rope_partial,rotation,window
loc: 634