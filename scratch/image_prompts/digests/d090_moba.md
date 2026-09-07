# d090 moba  [attention]
aliases: moba,moba_attention
file: Codes/framework/attentions/moba_attention.py  doc: Docs/content/attentions/moba_attention.md
class LoRALinear(Module)
class MoBAAttention(BaseAttention)
  doc: MoBA: Mixture of Block Attention for Long-Context Modeling. | Hardened for production across diverse architectures (DistilBERT, RoBERTa, GPT-2, Qwen3.5).
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, block_size, top_k, rank, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
  target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
  Q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
  K = self.k_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2)
  V = self.v_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  attn_probs = F.softmax(scores, dim=-1).to(dtype)
  head_mask = head_mask.view(1, -1, 1, 1)
shapes: [B, H, nb, nb [B, L, H, d [nb, nb [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rotation,topk
loc: 604