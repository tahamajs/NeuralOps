# d070 inject_vspec_attn  [attention]
aliases: inject_vspec_attn
file: Codes/framework/attentions/vspec.py  doc: NONE
doc: VSpec++ Attention: Variational Spectral Token-Adaptive Attention. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class VariationalSpectralGate(Module)
  doc: Input-dependent variational gate over truncated singular values with masked pooling.
class HeadGate(Module)
  doc: Learned per-example head importance scalars with masked pooling.
class TokenRouter(Module)
  doc: Top-k token selector based on learned saliency scores with guaranteed [CLS] retention.
class DynamicLowRankProjector(Module)
  doc: SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters.
class VSpecAttention(BaseAttention)
  doc: VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module.
__init__(hidden_dim, rank)
forward(x, valid_mask)
__init__(hidden_dim, n_heads)
forward(x, valid_mask)
__init__(hidden_dim, keep_ratio, min_tokens)
forward(x, valid_mask, is_decoder)
__init__(rank, in_dim, out_dim, bias, lora_rank, lora_alpha, freeze_base)
forward(x, gate)
__init__(config, rank, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  mu, log_sigma = stats.chunk(2, dim=-1)
  scores = scores.masked_fill(~valid_mask, -10000.0)
  scores = scores.masked_fill(~causal_mask[0], -10000.0)
  routed = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
  z = torch.matmul(x, self.V)
  out = torch.matmul(z, self.U.t())
  Q = Q.view(B, k_len, H, Dh).transpose(1, 2)
  K = K.view(B, k_len, H_kv, Dh).transpose(1, 2)
shapes: [B, N [w, b
parameters: torch.randn(out_dim, rank | torch.ones(rank | torch.randn(in_dim, rank | torch.zeros(out_dim | torch.empty(lora_rank, in_dim
features: adaptive,causal,chunked,distill,gating,gqa,lora,lowrank,peft,pool,rank,rope,rotation,svd
loc: 733