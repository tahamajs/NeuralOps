# d097 oha  [attention]
aliases: oha,optimized_hybrid_attention
file: Codes/framework/attentions/optimized_hybrid_attention.py  doc: Docs/content/attentions/optimized_hybrid_attention.md
doc: Optimized Hybrid Attention (OHA) – Production-Hardened. | Fuses localized sliding-window self-attention with sequence-compressed low-rank global attention | via dynamic head-wise gated modulation and exact pretrained weight transfer. | Supports:
class LoRALinear(Module)
class OptimizedHybridAttention(BaseAttention)
  doc: Optimized Hybrid Attention (OHA) combining sliding-window local and compressed global attention.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, window_size, compression_rank, use_gate, gate_bias, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
  K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
  V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
  local_scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  local_scores = local_scores.masked_fill(~local_mask, -1e4)
  local_probs = F.softmax(local_scores, dim=-1).to(dtype)
  local_ctx = torch.matmul(local_probs, V)
shapes: [B, N, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation,window
loc: 540