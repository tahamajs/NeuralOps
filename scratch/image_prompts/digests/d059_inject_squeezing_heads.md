# d059 inject_squeezing_heads  [attention]
aliases: inject_squeezing_heads,inject_squeezing_heads_distill
file: Codes/framework/attentions/squeezing_heads_distill.py  doc: NONE
doc: Squeezing Heads Attention Distillation Layer. | Multi-head attention wrapper designed for knowledge distillation and head capacity compression experiments. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class SqueezingHeadsDistiller(BaseAttention)
  doc: Attention layer wrapper for multi-head distillation and compression studies. | Hardened for production use across diverse transformer architectures.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, old_attn, rank, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
  q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.k_proj(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,distill,gqa,lora,pool,rank,rope,rope_partial,rotation,svd
loc: 608