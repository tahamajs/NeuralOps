# d125 svd  [attention]
aliases: svd,svd_sequence,svd_sequence_attention
file: Codes/framework/attentions/svd_attention.py  doc: NONE
doc: Low-Rank SVD-Compressed Sequence Attention Mechanism. | Reduces quadratic sequence complexity O(N^2) to linear in sequence length O(N * r) | by projecting Key and Value sequence dimensions into the top-r left-singular subspace | of the sequence representation.
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class SVDLinearAttention(BaseAttention)
  doc: Low-Rank SVD-Compressed Sequence Attention Mechanism. | Hardened for production use across diverse transformer architectures.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, old_attn, rank, lora_rank, lora_alpha, is_decoder, dropout)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_pooled = b.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
  K = self.key(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  V = self.value(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
  probs = F.softmax(scores, dim=-1).to(dtype)
shapes: [B, H, N, d
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,distill,gqa,lora,lowrank,pool,rank,rope,rope_partial,rotation,svd
loc: 660