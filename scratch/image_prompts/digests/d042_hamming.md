# d042 hamming  [attention]
aliases: hamming,hamming_attention
file: Codes/framework/attentions/hamming_attention.py  doc: Docs/content/attentions/hamming_attention.md
doc: Hamming Attention Distillation (HAD) Module. | Approximates dense inner products with differentiable continuous relaxations of Hamming distances | over sign-binarized representations with calibrated temperature scaling, multi-architecture weight transfer, | trainable LoRA residual adapters, GQA, RoPE, and causal decoding.
class LoRALinear(Module)
class HammingAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, distill_temperature, sparsity_ratio, lora_rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  Q = self.q_proj(hidden_states).view(B, L_q, H, Dh).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, L_k, self.num_kv_heads, Dh).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, L_k, self.num_kv_heads, Dh).transpose(1, 2)
  q_b = torch.tanh(5.0 * Q * self.q_scale.view(1, H, 1, 1))
  k_b = torch.tanh(5.0 * K * self.k_scale.view(1, H, 1, 1))
  scores = self.temp * (torch.matmul(q_b, k_b.transpose(-2, -1)) - Dh) / (2.0 * math.sqrt(Dh))
  scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
  scores = scores.masked_fill(~causal_tril, -10000.0)
shapes: [1, H, 1, 1 [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.ones(self.num_heads, 1
features: calib,causal,distill,gqa,lora,peft,rank,rope,rotation,sparsity
loc: 588