# d132 template_attention  [attention]
aliases: low_rank_modulated_attention,template_attention,your_attention_name,your_custom_attention
file: Codes/framework/attentions/template_attention.py  doc: Docs/content/attentions/template_attention.md
doc: Template Attention Mechanism (YourCustomAttention). | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class YourCustomAttention(BaseAttention)
  doc: Template attention layer with optional low-rank pooling modulation.
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.key(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.value(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
  scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  modulation = 1.0 + torch.tanh(H_rank).mean(dim=-1, keepdim=True).view(batch_size, 1, 1, 1)
  attn_weights = F.softmax(scores, dim=-1)
  head_mask = head_mask.view(1, -1, 1, 1)
  head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
features: causal,distill,gating,gqa,lora,lowrank,pool,rank,rope,rope_partial,rotation
loc: 581