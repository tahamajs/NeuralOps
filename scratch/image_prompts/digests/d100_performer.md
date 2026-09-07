# d100 performer  [attention]
aliases: performer,performer_attention
file: Codes/framework/attentions/performer_attention.py  doc: Docs/content/attentions/performer_attention.md
doc: Performer Attention – Fast Attention via Positive Orthogonal Random Features (FAVOR+). | Production-hardened with full multi-architecture support. | Supports: | - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
class LoRALinear(Module)
class PerformerAttention(BaseAttention)
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, eps, pretrained_attn, old_attn, comp_config)
forward(hidden_states, attention_mask, head_mask, position_embeddings, encoder_hidden_states)
tensor_ops:
  return W.t().contiguous()
  proj = torch.matmul(x, self.omega.to(dtype=x.dtype, device=x.device).t())
  w = w.reshape(w.size(0), -1)
  Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2) * scale
  K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2) * scale
  V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
  kv_outer = torch.einsum("bhnm,bhnd->bhnmd", K_feat.float(), V.float())
shapes: [B, N, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation
loc: 667