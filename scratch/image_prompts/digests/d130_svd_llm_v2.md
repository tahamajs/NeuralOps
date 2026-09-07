# d130 svd_llm_v2  [attention]
aliases: svd_llm_v2,svd_llm_v2_attention
file: Codes/framework/attentions/svd_llm_v2_attention.py  doc: Docs/content/attentions/svd_llm_v2_attention.md
doc: SVD-LLM V2 Attention: Truncation-Aware SVD for LLM Compression. | Direct low-rank factorized parameterization maintaining explicit U, Sigma, and V^H factor buffers, | with trainable LoRA adapters for recovery fine-tuning. | Hardened for production use across diverse transformer architectures:
class SVDLLMV2Attention(BaseAttention)
  doc: SVD-factorized attention layer with explicit U, Sigma, V^H matrix buffers | and trainable LoRA recovery adapters.
__init__(config, old_attn, rank, energy_threshold, lora_rank, lora_alpha, is_decoder, dropout)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  Q = Q_act.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
  K = K_act.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
parameters: torch.empty(self.q_head_size, self.rank | torch.empty(self.rank | torch.empty(self.rank, self.hidden_size | torch.zeros(self.q_head_size | torch.empty(self.kv_head_size, self.rank
features: causal,distill,gqa,lora,peft,pool,rank,rope,rope_partial,rotation,svd
loc: 631