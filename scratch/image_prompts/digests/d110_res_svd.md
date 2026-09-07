# d110 res_svd  [attention]
aliases: res_svd,spectral_residual_svd
file: Codes/framework/attentions/res_svd.py  doc: Docs/content/attentions/res_svd.md
doc: Spectral Residual Low-Rank SVD Attention (Res-SVD Attention) – Production-Hardened. | Combines base truncated SVD low-rank factorizations with trainable residual adapters: | W = U_base V_base^T + alpha * B A | where B A is initialized directly from the top-k residual singular directions of W - W_base.
class ResSVDAttention(BaseAttention)
__init__(config, base_rank, residual_rank, rank, scaling, factorize_base, freeze_base, dropout, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.query(hidden_states).view(B, N, H, D).permute(0, 2, 1, 3)
  K = self.key(hidden_states).view(B, N, self.num_kv_heads, D).permute(0, 2, 1, 3)
  V = self.value(hidden_states).view(B, N, self.num_kv_heads, D).permute(0, 2, 1, 3)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
  attn_weights = attn_weights * head_mask.view(1, self.num_heads, 1, 1).to(attn_weights.dtype)
  attn_weights = attn_weights * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(attn_weights.dtype)
  context = torch.matmul(attn_weights, V).permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
shapes: [B, N, H, D [N, N
features: causal,distill,gqa,lora,peft,rank,rope,rotation,svd
loc: 485