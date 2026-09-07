# d038 gqa  [attention]
aliases: gqa,grouped_query_attention
file: Codes/framework/attentions/gqa_attention.py  doc: NONE
doc: Grouped-Query Attention (GQA). | Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by dividing | $H_Q$ query heads into $G$ groups, where each group shares a single Key-Value head pair ($H_{KV} = G \le H_Q$).
class LoRALinear(Module)
class GroupedQueryAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, dim, num_heads, num_kv_heads, lora_rank, lora_alpha, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
  attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
  head_mask = head_mask.view(1, -1, 1, 1)
shapes: [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 575