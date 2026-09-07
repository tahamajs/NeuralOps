# d134 token_compressed  [attention]
aliases: token_compressed,token_compressed_attention,token_compressed_module
file: Codes/framework/attentions/token_compressed_attention.py  doc: Docs/content/attentions/token_compressed.md
doc: Token-Compressed Attention. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class LoRALinear(Module)
  doc: Linear projection with frozen base weights and trainable LoRA adapters.
class TokenCompressedAttention(BaseAttention)
  doc: Token-Compressed Attention Module. | Compresses Key and Value sequence tokens using energy/norm selection while preserving [CLS] (index 0).
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, compression_ratio, pretrained_attn, lora_rank, lora_alpha, comp_config, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  top_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
  total_importance = total_importance.masked_fill(mask == 0, -10000.0)
  total_importance = total_importance.masked_fill(~valid_mask, -10000.0)
  top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H_kv, -1, Dh)
  top_idx_4d = top_indices.unsqueeze(1).unsqueeze(2).expand(-1, mask.shape[1], mask.shape[2], -1)
  q = self.q_lin(hidden_states).view(B, L, H, Dh).transpose(1, 2)
  k = self.k_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2)
  v = self.v_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2)
shapes: [B, L [B, L, H * Dh [B, L, H, Dh
parameters: torch.empty(out_features, in_features | torch.empty(out_features | torch.empty(lora_rank, in_features | torch.empty(out_features, lora_rank
formulas/comments:
  Query token i can only attend to compressed keys whose original position <= i
features: causal,distill,gqa,lora,peft,rank,rope,rope_partial,rotation,topk
loc: 727