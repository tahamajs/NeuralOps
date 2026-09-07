# d074 kda  [attention]
aliases: kda,kda_attention
file: Codes/framework/attentions/kda_attention.py  doc: Docs/content/attentions/kda_attention.md
doc: Kernelized Decay Attention (KDA) / Recurrent Linear Attention (Hardened). | Implements multi-head linear recurrence with learned exponential decay gating, | spherical normalization, denominator tracking, and pretrained weight transfer. | Supports:
class LoRALinear(Module)
class KDAAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, d_model, num_heads, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  attn_intra = (qc @ kc.transpose(-1, -2)) * D_intra
  chunk_S = k_decayed.transpose(-1, -2) @ vc
  chunk_Z = k_decayed.transpose(-1, -2) @ torch.ones_like(vc[..., :1])
  attn_scores = (q.float() @ k.float().transpose(-1, -2))  # [B, H, L, L]
  hidden_states = hidden_states.contiguous()
  q_raw = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
  k_raw = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)
shapes: [1, H, 1, 1 [B, H, L, L [B, L, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.ones(self.num_heads, 1, 1
features: causal,chunked,decay,distill,gqa,lora,peft,rank,rope,rotation
loc: 717