# d075 leka_v1  [attention]
aliases: leka_v1
file: Codes/framework/attentions/leka_v1.py  doc: Docs/content/attentions/leka_v1.md
doc: LEKAv1: Linear-time Efficient Kernel Attention (Production-Hardened). | Mathematical Mechanism: | Approximates the Gaussian RBF kernel via Random Fourier Feature (RFF) mapping: | Supports:
class LoRALinear(Module)
class LEKAV1Attention(BaseAttention)
  doc: LEKAv1: Linear-time Efficient Kernel Attention (Hardened).
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(config, comp_config, layer_idx, total_layers, bert_mode, rank, pretrained_attn, old_attn)
forward(hidden_states, attention_mask, head_mask, position_embeddings, output_attentions)
tensor_ops:
  features = torch.einsum("bhnd,hdf->bhnf", x_norm, self.proj.to(device=x.device, dtype=x.dtype))
  w = w.reshape(w.size(0), -1)
  q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
  k = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  v = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2)
  kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_features.float(), v.float())
  num = torch.einsum("bhnf,bhnfd->bhnd", Q_features.float(), KV_cum)
  denom = torch.einsum("bhnf,bhnf->bhn", Q_features.float(), z_cum).unsqueeze(-1).clamp(min=1e-5)
shapes: [B, N, H, d
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor(2.0
features: causal,distill,gating,gqa,lora,peft,rank,rope,rotation
loc: 540