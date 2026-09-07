# d012 aegis_v6  [attention]
aliases: aegis_v6,aegis_v6_attn
file: Codes/framework/attentions/aegis_v6.py  doc: Docs/content/attentions/aegis_v6.md
doc: AEGIS-Attention V6: Unified Hybrid Transformer Attention Module. | Combines SVD low-rank factorized projections with LoRA adapters, | sliding window attention, global linear kernel attention with adaptive spectral gating, | and context fusion initialized to prioritize high-fidelity local attention.
class AegisConfigV3()
class FactorisedLowRankProjection(Module)
  doc: Low-rank SVD projection with LoRA update branch: | 88291y = x (V \Sigma U^T) + x (B^T A^T) + b88291
class KernelFeatureConfig()
class StableKernelFeatureMap(Module)
class GatingModule(Module)
class ContextFusionGate(Module)
class AegisAttentionV3(BaseAttention)
__init__()
__init__(in_features, out_features, svd_rank, lora_rank, pretrained_weight, pretrained_bias)
forward(x)
__init__(config)
forward(x)
__init__(hidden_size, num_heads, head_dim, use_query, use_input)
forward(hidden_states, Q)
__init__(hidden_size, num_heads, head_dim, init_bias)
forward(hidden_states, local_out, global_out)
__init__(config, pretrained_qkv, pretrained_attn, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  W = W.view(self.out_features, rep, -1).mean(dim=1)
  bias = bias.view(self.out_features, rep).mean(dim=1)
  xV = torch.matmul(x_f, self.V.float())
  out_svd = torch.matmul(xV_S, self.U.float().t())
  xB = torch.matmul(x_f, self.B.float().t())
  out_lora = torch.matmul(xB, self.A.float().t())
  phi = torch.matmul(x, self.proj_buf.to(x.dtype))
  in_gate = torch.sigmoid(logits).permute(0, 2, 1).unsqueeze(-1)
shapes: [B, N, h, d [d, d [w, b
parameters: torch.ones(self.svd_rank, device=device, dtype=dtype | torch.zeros(out_features, lora_rank, device=device, dtype=dt | torch.zeros(lora_rank, in_features, device=device, dtype=dty | torch.zeros(out_features, device=device, dtype=dtype
features: adaptive,causal,distill,gating,gqa,lora,lowrank,rank,rope,rotation,svd,window
loc: 807