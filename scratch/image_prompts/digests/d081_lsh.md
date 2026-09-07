# d081 lsh  [attention]
aliases: lsh,lsh_attention
file: Codes/framework/attentions/lsh_attention.py  doc: Docs/content/attentions/lsh_attention.md
doc: Locality-Sensitive Hashing (LSH) Attention – Reformer style. | Approximates dense attention by clustering keys and queries into discrete hash buckets | via angular random hyperplane projections with fallback context, multi-architecture weight transfer, | trainable LoRA residual adapters, GQA, RoPE, and causal decoding.
class LoRALinear(Module)
class LSHAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, rounds, num_buckets, lora_rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  dots = torch.matmul(norm, proj)
  Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
  attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
shapes: [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,cluster,distill,gqa,hash,lora,rank,rope,rotation
loc: 489