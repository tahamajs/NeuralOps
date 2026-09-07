# d117 soa_attn  [attention]
aliases: soa_attn,soa_module
file: Codes/framework/attentions/soa_attn.py  doc: Docs/content/attentions/soa_attn.md
doc: Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class SOAAttention(BaseAttention)
  doc: Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks. | Factorizes dense Q, K, and V linear projections into low-rank bottleneck pairs
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, model_config, rank, k_rank, dropout, pretrained_attn, old_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = q.view(B, N, H, D).transpose(1, 2)
  k = k.view(B, N, H_kv, D).transpose(1, 2)
  v = v.view(B, N, H_kv, D).transpose(1, 2)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  attn_weights = F.softmax(scores, dim=-1)
  head_mask = head_mask.view(1, -1, 1, 1)
  head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
  context = torch.matmul(attn_weights, v)
shapes: [B, N, H * D [B, N, H, D [N, N
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rope_partial,rotation,svd
loc: 525