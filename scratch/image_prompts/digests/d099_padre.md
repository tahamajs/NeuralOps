# d099 padre  [attention]
aliases: padre,padre_attention
file: Codes/framework/compression_engine/padre_attention.py  doc: Docs/content/compressions/padre_attention.md
doc: PADRe: Polynomial Attention with Dropout Regularization. | Replaces softmax with a stable polynomial approximation, enabling faster | inference and kernelized attention across diverse transformer architectures. | Production-Hardened with:
class LoRALinear(Module)
  doc: Linear layer augmented with optional trainable low-rank adaptation.
class PADReAttention(BaseAttention)
  doc: Polynomial Attention with Dropout Regularization (PADRe). | Replaces softmax with centered, numerically stable polynomial kernel (1 + S/order)^order.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, order, temperature, pretrained_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(q, k.transpose(-2, -1)) * scale
  masked_for_max = scores.masked_fill(invalid_mask, -1e9)
  attn_weights = attn_weights.masked_fill(invalid_mask, 0.0)
  hm = hm.view(1, -1, 1, 1)
  hm = hm.view(hm.shape[0], 1, 1, 1)
shapes: [1, 1, N, N [B, 1, 1, N [B, H, N, N [B, N
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: calib,causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 666