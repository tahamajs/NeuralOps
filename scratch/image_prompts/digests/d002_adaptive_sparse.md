# d002 adaptive_sparse  [attention]
aliases: adaptive_sparse,adaptive_sparse_attention
file: Codes/framework/attentions/adaptive_sparse_attention.py  doc: Docs/content/attentions/adaptive_sparse_attention.md
doc: Adaptive Sparse Attention Module (Production-Hardened). | Implements dynamic top-k and threshold-based adaptive sparse attention with LoRA adapters, | multi-architecture weight extraction, RoPE rotary embeddings, causal/bidirectional masking, | and safe injection.
class LoRALinear(Module)
class AdaptiveSparseAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, sparsity_ratio, dropout, lora_rank, comp_config, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  values = values.permute([2, 0, 1]).unsqueeze(0)
  Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  pos_ids = torch.arange(N_q, device=device).unsqueeze(0).expand(B, -1)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
shapes: [1, H, 1, 1 [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.full((self.num_heads,
features: adaptive,causal,distill,gqa,lora,peft,rank,rope,rotation,sparsity,topk
loc: 699