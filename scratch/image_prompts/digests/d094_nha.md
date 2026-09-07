# d094 nha  [attention]
aliases: nha,nha_attention,nha_compressor
file: Codes/framework/attentions/nha_attention.py  doc: Docs/content/attentions/nha_attention.md
doc: Native Hybrid Attention (NHA) – Production-Hardened. | Fuses localized sliding-window attention with global linear Random Fourier Features (RFF) | via dynamic token-level gating and exact pretrained weight transfer. | Supports:
class LoRALinear(Module)
class NativeHybridAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, window_size, rff_features, dropout, pretrained_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(band_mask.unsqueeze(0).unsqueeze(0), -1e4)
  scores = scores.masked_fill(pad_mask, -1e4)
  attn = F.softmax(scores, dim=-1).to(Q.dtype)
  return torch.matmul(attn, V)
  Qf = torch.matmul(Q_scaled, proj)
  Kf = torch.matmul(K_scaled, proj)
  KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)
shapes: [B, 1, N, 1 [B, N, 1
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation,window
loc: 708