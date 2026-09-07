# d071 inject_wgqa  [attention]
aliases: inject_wgqa
file: Codes/framework/attentions/wgqa_attention.py  doc: NONE
doc: Weighted Grouped-Query Attention (WGQA). | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class WeightedGroupedQueryAttention(BaseAttention)
  doc: Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance weights, | robust GQA pooling, LoRA adapters, RoPE support with fallback, and causal masking.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, lora_rank, lora_alpha, num_kv_heads, is_decoder, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w_k_or.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_val.view(self.num_kv_heads, rep, self.head_dim)
  .view(-1)
  w_v_or.view(self.num_kv_heads, rep, self.head_dim, -1)
  .view(self.kv_head_size, -1)
  b_val.view(self.num_kv_heads, rep, self.head_dim)
  .view(-1)
shapes: [B, 1, 1, 1 [B, H, 1, 1 [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.ones(self.num_kv_heads, 1
features: causal,distill,gqa,lora,pool,rank,rope,rotation
loc: 718