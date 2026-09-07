# d098 orion  [attention]
aliases: orion,orion_attention,orion_module
file: Codes/framework/attentions/orion.py  doc: Docs/content/attentions/orion.md
doc: ORION: Optimized Rectified Input-aware Orthogonal Network (Production-Hardened). | Hybrid multi-head attention architecture integrating zero-shot SVD low-rank factorized projections, | head-wise bifurcated linear kernel vs. rectified Softpick full attention, and random Fourier features. | Supports:
class ORIONConfig()
class FactorisedLinear(Module)
  doc: Low-rank factorized linear layer with SVD decomposition and LoRA adapters.
class ORIONAttention(BaseAttention)
  doc: ORION hybrid multi-head attention module.
__init__(in_features, out_features, bias, energy, min_rank, max_rank, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, rank, model_config, orion_cfg, pretrained_attn, old_attn, comp_config, bert_mode)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  return torch.matmul(self.up.weight, self.down.weight)
  w = w.reshape(w.size(0), -1)
  Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, proj).clamp(max=11.0)
  K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, proj).clamp(max=11.0)
  kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_feat, V)
  num = torch.einsum("bhnf,bhnfd->bhnd", Q_feat, KV_cum)
  denom = torch.einsum("bhnf,bhnf->bhn", Q_feat, z_cum).unsqueeze(-1).clamp(min=1e-5)
  KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
shapes: [B, L, H * Dh
parameters: torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank
features: causal,distill,gqa,lora,peft,rank,rope,rotation,svd
loc: 638