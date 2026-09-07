# d128 svd_linear  [attention]
aliases: svd_linear,svd_linear_attention,svd_linear_attn
file: Codes/framework/attentions/svd_linear_attention.py  doc: Docs/content/compressions/svd_linear.md
doc: SVD Factorized Linear Attention & Kernelized Linear Attention Modules with Exponential Decay Gating. | Hardened for production use across diverse transformer architectures: | - DistilBERT (Flat MHA) | - RoBERTa / BERT (Nested Encoder Self-Attention)
class SVDLinear(Module)
  doc: Low-Rank Factorized Linear Layer with SVD bases and trainable LoRA adapters: | $$y = (x V^\top) U^\top + b + \frac{\alpha}{r_{\text{lora}}} (x A^\top) B^\top$$
class SVDLinearAttention(BaseAttention)
  doc: SVD Factorized Linear Attention & Kernelized Linear Attention with Exponential Decay Gating. | Hardened for production use across diverse transformer architectures.
__init__(in_features, out_features, rank, bias, freeze_svd, lora_rank, lora_alpha, use_fused_kernel, use_rms_norm, eps, dtype, device)
forward(x)
__init__(config, old_attn, rank, mode, freeze_svd, lora_rank, lora_alpha, is_decoder, use_decay_gating, decay_activation, decay_channel_wise, use_fused_kernel)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  W = torch.matmul(self.U, self.V)
  W = W + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
  U_orth = torch.matmul(self.U.t(), self.U) - torch.eye(self.rank, device=self.U.device, dtype=self.U.dtype)
  V_orth = torch.matmul(self.V, self.V.t()) - torch.eye(self.rank, device=self.V.device, dtype=self.V.dtype)
  chunks_w = in_proj_w.chunk(3, dim=0)
  chunks_b = in_proj_b.chunk(3, dim=0)
  w = w.contiguous().reshape(self.num_kv_heads, rep, -1).mean(dim=1)
  b = b.contiguous().reshape(self.num_kv_heads, rep, -1).mean(dim=1).reshape(-1)
shapes: [B, 1, N, 1 [B, H, N [B, H, N, 1 [B, H, N, 1, 1 [B, H, N, N
parameters: torch.empty(rank, in_features, dtype=dtype, device=device | torch.empty(out_features, rank, dtype=dtype, device=device | torch.ones(rank, dtype=dtype, device=device | torch.empty(self.lora_rank, in_features, dtype=dtype, device | torch.empty(out_features, self.lora_rank, dtype=dtype, devic
formulas/comments:
  Pairwise log decay matrix: log_M[t, j] = log_gamma[t] - log_gamma[j] for j <= t
  Affinity: A = phi_q @ phi_k^T  [B, H, N, N]
features: causal,chunked,decay,distill,gating,gqa,lora,lowrank,peft,pool,rank,rope,rope_partial,rotation
loc: 1030