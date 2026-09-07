# d030 full_attention  [attention]
aliases: full_attention
file: Codes/framework/attentions/full_attention.py  doc: Docs/content/attentions/full_attention.md
doc: Standard Multi-Head Scaled Dot-Product Attention (Vaswani et al., 2017). | Computes pairwise token interactions across multiple representation subspaces with frozen base weights, | trainable LoRA residual adapters, multi-architecture weight transfer, GQA, RoPE, and causal decoding.
class LoRALinear(Module)
class FullAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, lora_rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
  attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
  head_mask = head_mask.view(1, -1, 1, 1)
shapes: [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rotation
loc: 464