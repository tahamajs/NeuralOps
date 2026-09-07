# d020 compressed_mha  [attention]
aliases: compressed_mha,compressed_mha_attention
file: Codes/framework/attentions/compressed_mha_attention.py  doc: Docs/content/attentions/compressed_mha_attention.md
doc: Compressed Multi-Head Attention for DistilBERT and Multi-Architecture via Joint Truncated SVD. | Performs joint singular value decomposition on stacked query, key, and value projection matrices | as well as the final output projection matrix with frozen base parameters, trainable LoRA residual adapters, | GQA, RoPE, causal decoding, and full pretrained weight transfer.
class CompressedDistilBertAttention(BaseAttention)
__init__(model_config, rank, rank_qkv, rank_out, dropout, lora_rank, pretrained_attn, config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  return (self.U_qkv * self.S_qkv.view(1, -1)) @ self.Vh_qkv
  return (self.U_out * self.S_out.view(1, -1)) @ self.Vh_out
  proj = torch.matmul(hidden_states, self.Vh_qkv.t()) * self.S_qkv.unsqueeze(0).unsqueeze(0)
  qkv_comp = torch.matmul(proj, self.U_qkv.t())
  q_raw, k_raw, v_raw = torch.split(qkv_comp, [self.all_head_size, self.kv_head_size, self.kv_head_size], dim=-1)
  q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
  k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
shapes: [N, N [w, b
parameters: torch.randn(qkv_out_dim, self.rank_qkv | torch.ones(self.rank_qkv | torch.randn(self.rank_qkv, self.hidden_size | torch.randn(self.hidden_size, self.rank_out | torch.ones(self.rank_out
features: causal,distill,gqa,lora,rank,rope,rotation,svd
loc: 479