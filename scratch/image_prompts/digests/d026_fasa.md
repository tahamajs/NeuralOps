# d026 fasa  [attention]
aliases: fasa,fasa_attention,mixattention_moska
file: Codes/framework/attentions/mixattention_moska.py  doc: Docs/content/attentions/mixattention_moska.md
doc: MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA). | Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
class LoRALinear(Module)
class MixAttentionMoSKA(BaseAttention)
  doc: MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA). | Prunes key-value tokens based on variance importance while strictly protecting [CLS] (index 0).
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, fasa_keep_ratio, pretrained_attn, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
  target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
  importance = importance.masked_fill(~mask_2d, -10000.0)
  importance = importance.masked_fill(~mask_2d, -10000.0)
  idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
  q = self.W_q(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
  k = self.W_k(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
  v = self.W_v(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
shapes: [B, L [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,rank,rope,rotation,sparsity,topk
loc: 579