# d089 mla  [attention]
aliases: mla,mla_attention
file: Codes/framework/attentions/mla_attention.py  doc: Docs/content/attentions/mla_attention.md
doc: Multi-Head Latent Attention (MLA) – DeepSeek-V2. | Paper: https://arxiv.org/abs/2405.04434 | Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
class LoRALinear(Module)
class MLAAttention(BaseAttention)
  doc: Multi-Head Latent Attention (MLA) Mechanism. | Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w_k_f = w_k_f.view(self.kv_dim, rep, -1).mean(dim=1)
  w_v_f = w_v_f.view(self.kv_dim, rep, -1).mean(dim=1)
  q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_up(latent).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.v_up(latent).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  attn_probs = F.softmax(scores, dim=-1).to(dtype)
  head_mask = head_mask.view(1, -1, 1, 1)
shapes: [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rotation,svd
loc: 511