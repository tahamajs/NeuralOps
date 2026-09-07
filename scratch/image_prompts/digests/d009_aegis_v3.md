# d009 aegis_v3  [attention]
aliases: aegis_v3,aegis_v3_module,aegisattention
file: Codes/framework/attentions/aegis_v3.py  doc: Docs/content/attentions/aegis_v3.md
doc: AEGIS-Attention V3: Adaptive Enhanced Global-Local Integrated Sparse Attention Baseline. | Combines SVD-factorized projection matrices with energy-based rank selection and LoRA adapters, | split-head local sliding-window attention with smooth activations, | and global linear kernel attention with adaptive gating.
class AegisConfig()
  doc: Configuration parameters for AegisAttention V3.
class FactorisedLinear(Module)
  doc: SVD-factorized linear layer with energy-based rank truncation and trainable LoRA adapters. | Base SVD weights are frozen; LoRA branch provides trainable parameters.
class AegisAttention(BaseAttention)
__init__(in_features_or_weight, out_features_or_bias, energy, min_rank, max_rank, rank, bias, lora_rank, lora_alpha)
forward(x)
__init__(model_config, config, aegis_cfg, pretrained_attn, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class DummyConfig()
tensor_ops:
  W = W.view(self.out_features, rep, -1).mean(dim=1)
  bias = bias.view(self.out_features, rep).mean(dim=1)
  base_w = torch.matmul(self.up.weight, self.down.weight)
  scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
  scores = scores.masked_fill(~mask, mask_val)
  scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), mask_val)
  return torch.matmul(attn, V)
  Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj_f).clamp(max=max_clamp)
shapes: [B, N, H * D [B, N, H, D [w, b
parameters: torch.randn(lora_rank, in_dim, device=device, dtype=dtype | torch.zeros(out_dim, lora_rank, device=device, dtype=dtype | torch.randn(lora_rank, in_dim | torch.zeros(out_dim, lora_rank | rff_init
features: adaptive,causal,distill,gating,gqa,lora,rank,rope,rotation,sparsity,svd,window
loc: 729