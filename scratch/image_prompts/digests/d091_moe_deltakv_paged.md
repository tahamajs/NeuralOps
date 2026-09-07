# d091 moe_deltakv_paged  [attention]
aliases: moe_deltakv_paged
file: Codes/framework/attentions/moe_deltakv_paged.py  doc: Docs/content/attentions/moe_deltakv_paged.md
doc: MoE-DeltaKV-Paged Hybrid Attention Module. | Combines Mixture-of-Block-Attention (MoBA) with Paged Delta-KV cache compression. | Fixes: | - Straight-Through Estimator (STE) for differentiable 4-bit uniform and SVD quantization.
class MoEDeltaKVPagedAttention(Module)
  doc: MoE-DeltaKV-Paged Hybrid Attention Module. | Combines Mixture-of-Block-Attention (MoBA) with Paged Delta-KV cache compression.
__init__(config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  delta_reshaped = delta.reshape(-1, d)
  delta_approx = delta_approx.reshape(B, H, N, d)
  q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
  k = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
  v = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)
  scores = torch.matmul(q, k_hat.transpose(-2, -1)) * scale
  scores = scores.masked_fill(pad_mask, -10000.0)
  attn = F.softmax(scores, dim=-1)
shapes: [B, H, N, d [B, N, H, d [w, b
features: causal,distill,gqa,lowrank,moe,quant,rank,rope,rotation,svd,topk
loc: 468