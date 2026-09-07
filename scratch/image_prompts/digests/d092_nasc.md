# d092 nasc  [attention]
aliases: nasc,nasc_attention
file: Codes/framework/attentions/nasc_attention.py  doc: Docs/content/attentions/nasc_attention.md
doc: NASC: NEXUS-AEGIS Spectral Cascade Attention. | Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
class LoRALinear(Module)
class NASCAttention(BaseAttention)
  doc: NASC: NEXUS-AEGIS Spectral Cascade Attention. | Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, window_size, block_size, top_k, rff_features, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
  target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -10000.0)
  scores = scores.masked_fill(pad_k, -10000.0)
  attn = F.softmax(scores, dim=-1).to(Q.dtype)
  return torch.matmul(attn, V)
  Qb = Q_pad.view(B, H, nb, self.block_size, d)
shapes: [B, 1, 1, 3 [B, H, L, d [B, L, H, d [nb, nb [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.randn(self.head_dim, rff_features
features: causal,distill,gating,gqa,lora,rank,rope,rotation,sparsity,topk,window
loc: 651