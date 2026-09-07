# d060 inject_svd_compressed  [attention]
aliases: inject_svd_compressed,inject_svd_compressed_attention
file: Codes/framework/attentions/svd_compressed.py  doc: NONE
doc: SVD-Compressed Attention with Importance-Aware Lagrangian Rank Allocation. | Truncated Singular Value Decomposition (SVD) attention compression module with | Lagrangian spectral-entropy optimal rank budgeting and optional low-rank plus sparse residual decomposition. | Mathematical Mechanism:
class SVDCompressedAttention(BaseAttention)
  doc: Multi-head attention with truncated SVD projections, trainable LoRA adapters, | GQA support, RoPE support, and optional sparse residuals.
__init__(config, old_attn, dropout, layer_idx, ranks, lora_rank, lora_alpha, is_decoder, add_sparse, sparsity_level)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w = w.t().contiguous()
  return w.t().contiguous()
  w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
parameters: torch.empty(self.lora_rank, self.hidden_size | torch.empty(self.q_head_size, self.lora_rank | torch.empty(self.lora_rank, self.hidden_size | torch.empty(self.kv_head_size, self.lora_rank | torch.empty(self.lora_rank, self.hidden_size
formulas/comments:
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
  =============================================================================
features: causal,decay,distill,entropy,gqa,lora,lowrank,peft,pool,rank,rope,rope_partial,rotation,sparsity
loc: 1074