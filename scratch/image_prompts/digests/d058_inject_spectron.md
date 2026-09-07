# d058 inject_spectron  [attention]
aliases: inject_spectron,inject_spectron_attention
file: Codes/framework/attentions/spectron.py  doc: NONE
doc: SpectronAttention: Importance-Aware Low-Rank + Sparse Attention. | Dynamically allocates non-uniform low-rank capacities across attention projection matrices under a global parameter | budget via Lagrangian spectral entropy optimization, optionally augmented by trainable magnitude-pruned sparse residuals. | Mathematical Mechanism:
class LowRankSparseLinear(Module)
  doc: Composite linear layer: $y = (x W_{\text{down}}^T) W_{\text{up}}^T + x S^T + b$.
class SpectronAttention(BaseAttention)
  doc: Importance-aware SVD-compressed attention module with optional trainable sparse residual branch, | trainable LoRA adapters, GQA support, RoPE support, and causal masking.
class SparsePruningCallback(TrainerCallback)
  doc: Trainer callback periodically magnitude-pruning sparse residual parameters.
__init__(down, up, sparse_weight, bias)
forward(x)
__init__(config, old_attn, dropout, layer_idx, ranks, lora_rank, lora_alpha, is_decoder, add_sparse, sparsity_level)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
__init__(model, prune_freq, final_sparsity)
tensor_ops:
  return torch.matmul(self.up.weight, self.down.weight) + self.sparse_weight
  threshold = torch.kthvalue(w.view(-1), w.numel() - k).values
  w = w.t().contiguous()
  return w.t().contiguous()
  w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
parameters: sparse_weight | torch.empty(self.lora_rank, self.hidden_size | torch.empty(self.q_head_size, self.lora_rank | torch.empty(self.lora_rank, self.hidden_size | torch.empty(self.kv_head_size, self.lora_rank
formulas/comments:
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
  ======================================================================
features: causal,decay,distill,entropy,gqa,lora,lowrank,peft,pool,rank,rope,rope_partial,rotation,sparsity
loc: 1074