# d013 asvd  [attention]
aliases: asvd,asvd_attention
file: Codes/framework/attentions/asvd_attention.py  doc: Docs/content/attentions/asvd_attention.md
doc: ASVD (Activation-Aware Singular Value Decomposition) Attention. | Paper: https://arxiv.org/abs/2307.07382 | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA)
class ASVDLinear(Module)
  doc: Low-rank factorized linear layer with explicit singular value scaling and trainable LoRA adapters: | $$y = ((x V^\top) \odot \Sigma) U^\top + (x A^\top) B^\top + b$$
class ASVDAttention(Module)
  doc: Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE, and LoRA. | Computational Complexity:
__init__(in_features, out_features, rank, bias, lora_rank, freeze_base, dtype, device)
forward(x)
__init__(config, rank, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  W = torch.matmul(self.U * self.S.unsqueeze(0), self.V)
  W = W + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
  q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
  k: torch.Tensor = self.key(hidden_states).view(B, N, kv_H, d).transpose(1, 2)
  v: torch.Tensor = self.value(hidden_states).view(B, N, kv_H, d).transpose(1, 2)
  scores: torch.Tensor = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
  extended_mask = extended_mask.masked_fill(~attention_mask, -10000.0)
  extended_mask = extended_mask.masked_fill(~attention_mask, -10000.0)
shapes: [B, N, D [B, N, H, d [N, N [w, b
parameters: torch.empty(out_features, rank, **factory_kwargs | torch.empty(rank, **factory_kwargs | torch.empty(rank, in_features, **factory_kwargs | torch.empty(out_features, **factory_kwargs | torch.empty(lora_rank, in_features, **factory_kwargs
features: causal,distill,gqa,lora,rank,rope,rotation,svd
loc: 712