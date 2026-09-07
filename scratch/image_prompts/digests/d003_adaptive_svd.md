# d003 adaptive_svd  [attention]
aliases: adaptive_svd,adaptive_svd_attention,adaptive_svd_attn
file: Codes/framework/attentions/adaptive_svd_attention.py  doc: Docs/content/attentions/adaptive_svd_attention.md
doc: Adaptive SVD Attention with Residual Adapters and Dynamic Energy Thresholding. | This module provides low-rank factorized linear projections for Multi-Head Attention, | decomposing full-rank weight matrices via dynamic cumulative energy-thresholded | Singular Value Decomposition (SVD):
class AdaptiveSVDLinear(Module)
  doc: Linear layer decomposed into a frozen SVD base and a trainable residual adapter. | Uses `lora_A` and `lora_B` parameter names so fine-tuning optimizers and LoRA filters
class AdaptiveSVDAttention(BaseAttention)
  doc: Multi-Head Attention with Adaptive SVD + Residual factorized projections.
__init__(in_features, out_features, svd_rank, residual_rank, bias, trainable_svd, dtype, device)
forward(x)
__init__(config, svd_rank, residual_rank, energy_threshold, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  W = torch.matmul(self.U, self.V)
  W = W + torch.matmul(self.lora_B, self.lora_A)
  values = values.permute([2, 0, 1]).unsqueeze(0)
  Q = self.query(hidden_states).view(B, N_q, H, D).transpose(1, 2)
  K = self.key(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2)
  V = self.value(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]
  scores = scores.masked_fill(~causal_tril, -10000.0)
shapes: [3 * D, D [w, b
parameters: torch.empty((out_features, self.svd_rank | torch.empty((self.svd_rank, in_features | torch.empty((out_features, self.residual_rank | torch.empty((self.residual_rank, in_features | torch.empty(out_features, **factory_kwargs
features: adaptive,causal,distill,gqa,lora,rank,rope,rotation,svd
loc: 672