# d080 lora_attn  [attention]
aliases: lora_attention,lora_attn
file: Codes/framework/attentions/lora_attention.py  doc: Docs/content/attentions/lora_attention.md
doc: LoRA Parameter-Efficient Fine-Tuning Attention Module. | Replaces standard query, key, value, and output linear projections with frozen pretrained base weights | and trainable low-rank residual adapters, with multi-architecture weight transfer, GQA, RoPE, and causal decoding.
class LoRALinear(Module)
class LoRAAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, model_config, rank, lora_rank, alpha, dropout, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
  attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
  head_mask = head_mask.view(1, -1, 1, 1)
shapes: [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rotation
loc: 448