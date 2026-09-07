# d126 svd-mamba  [attention]
aliases: svd-mamba,svd_mamba
file: Codes/framework/attentions/svd_mamba.py  doc: Docs/content/attentions/svd_mamba.md
doc: SVD-Mamba: SVD-factorized Selective State Space Model. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class SVDLinear(Module)
  doc: SVD-factorized linear layer with trainable LoRA adapters. | Base weight = U @ diag(S) @ V^T.
class SVDMamba(BaseAttention)
  doc: SVD-Mamba module with SVD-factorized projections, causal convolution, and selective SSM.
__init__(in_features, out_features, rank, lora_rank, lora_alpha, freeze_base)
forward(x)
__init__(config, rank, d_model, d_state, d_conv, expand, dt_rank, global_rank, lora_rank, lora_alpha, is_decoder, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  hidden_states = hidden_states.contiguous()
  x, z = xz.chunk(2, dim=-1)                # each [B, L, inner]
  x_t = x.transpose(1, 2)                   # [B, inner, L]
  x_conv = x_conv.transpose(1, 2)           # [B, L, inner]
  log_alpha = (delta.unsqueeze(-1) * A.view(1, 1, 1, -1)).float()
parameters: torch.randn(out_features, self.rank | torch.ones(self.rank | torch.randn(in_features, self.rank | torch.zeros(out_features | torch.empty(lora_rank, in_features
features: causal,chunked,decay,distill,gqa,lora,peft,rank,rope,ssm,svd
loc: 568