# d019 compressed_attention_layer_energy  [attention]
aliases: compressed_attention_layer_energy
file: Codes/framework/attentions/compressed_attention.py  doc: NONE
doc: Compressed Multi-Head Attention via Truncated Singular Value Decomposition (SVD). | Provides rank-constrained and singular-energy-thresholded low-rank approximations of standard | PyTorch attention input (QKV) and output projection matrices with full pretrained weight transfer, | trainable LoRA residual adapters, GQA, RoPE, and causal decoding support.
class CompressedAttentionLayer(BaseAttention)
class CompressedAttentionLayer2(CompressedAttentionLayer)
__init__(config, rank, rank_qkv, rank_out, lora_rank, original_layer, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  return (self.U_qkv * self.S_qkv_sqrt.view(1, -1)) @ (self.Vh_qkv * self.S_qkv_sqrt.view(-1, 1))
  return (self.U_out * self.S_out_sqrt.view(1, -1)) @ (self.Vh_out * self.S_out_sqrt.view(-1, 1))
  v_proj = torch.matmul(hidden_states, self.Vh_qkv.t()) * self.S_qkv_sqrt.view(1, 1, -1)
  qkv = torch.matmul(v_proj, (self.U_qkv * self.S_qkv_sqrt.view(1, -1)).t()) + self.bias_qkv
  q, k, v = torch.split(qkv, [self.all_head_size, self.kv_head_size, self.kv_head_size], dim=-1)
  q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
  k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
shapes: [N, N [w, b
parameters: torch.randn(self.rank_qkv, self.hidden_size | torch.ones(self.rank_qkv | torch.randn(qkv_out_dim, self.rank_qkv | torch.zeros(qkv_out_dim | torch.randn(self.rank_out, self.all_head_size
features: causal,distill,gqa,lora,rank,rope,rotation,svd
loc: 479