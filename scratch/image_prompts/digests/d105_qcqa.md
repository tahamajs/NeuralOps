# d105 qcqa  [attention]
aliases: qcqa,qcqa_attention
file: Codes/framework/attentions/qcqa_attention.py  doc: Docs/content/attentions/qcqa_attention.md
doc: Quality and Capacity-Aware Grouped Query Attention (QCQA) – Production-Hardened. | Dynamically synthesizes head-specific key-value representations for each query head | via continuous learned routing distributions over a compressed set of KV heads. | Supports:
class LoRALinear(Module)
class QCQAAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, num_kv_heads, router_temp, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
  k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
  v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
  q_t = q.permute(0, 2, 1, 3)
  k_t = k.permute(0, 2, 1, 3)
  q = q_rot.permute(0, 2, 1, 3)
  k = k_rot.permute(0, 2, 1, 3)
  route_weights = F.softmax(route_logits, dim=-1)
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 514