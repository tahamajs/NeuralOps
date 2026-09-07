# d004 adasplash  [attention]
aliases: adasplash,adasplash_attention
file: Codes/framework/attentions/adasplash_attention.py  doc: Docs/content/attentions/adasplash_attention.md
doc: AdaSplash Attention: Adaptive Splash Attention with α-Entmax Sparse Activations. | Combines localized sliding-window splash radius with sparse α-entmax adaptive thresholding | and global token routing, ensuring robust numerical convergence and exact pretrained weight transfer.
class AdaSplashLinear(Module)
  doc: Linear projection layer with frozen base weights and trainable residual LoRA adapter. | Uses `lora_A` and `lora_B` parameter names so recovery optimizers and LoRA filters
class AdaSplashAttention(BaseAttention)
  doc: AdaSplash Attention Layer with α-Entmax and Local Splash Radius. | Mathematical Formulation:
__init__(in_features, out_features, lora_rank, bias, dtype, device)
forward(x)
__init__(config, rank, alpha, n_iter, splash_radius, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  scores = scores.masked_fill(~valid_mask, -10000.0)
  sf = torch.nan_to_num(F.softmax(scores_clean.float(), dim=dim), nan=0.0).to(dtype)
  sf = sf.masked_fill(~valid_mask, 0.0)
  scores_valid = scores.masked_fill(~valid_mask, -10000.0)
  scaled_for_min = scaled.masked_fill(~valid_mask, 10000.0)
  p = p.masked_fill(~valid_mask, 0.0)
  p = p.masked_fill(~valid_mask, 0.0)
  scores_for_sm = scores.masked_fill(~valid_mask, -10000.0) if valid_mask is not None else scores
shapes: [3 * D, D [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((out_features, self.lora_rank | torch.empty((self.lora_rank, in_features
features: adaptive,causal,distill,gqa,lora,rank,rope,rotation,sparsity,window
loc: 752