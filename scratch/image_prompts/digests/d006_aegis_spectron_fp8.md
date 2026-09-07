# d006 aegis_spectron_fp8  [attention]
aliases: aegis_spectron_fp8,aegis_spectron_fp8_attn
file: Codes/framework/attentions/aegis_spectron_fp8.py  doc: Docs/content/attentions/aegis_spectron_fp8.md
doc: Aegis-Spectron-FP8 Hybrid Attention Module. | Integrates Aegis multi-branch local/global feature fusion with Spectron Lagrangian spectral entropy routing. | Dynamic routing measures singular value entropy per token to distribute token representation across: | 1. Local Sliding-Window Attention (O(N * W) with strict causal masking for decoders)
class AegisSpectronFP8Config()
  doc: Hyperparameter configuration dataclass for Aegis-Spectron-FP8.
class FactorizedLinear(Module)
  doc: Factorized linear projection with internal rank-r base factorization and trainable residual LoRA adapter: | Base: x -> lin1(x) [in_features -> rank] -> lin2(lin1(x)) [rank -> out_features] (frozen)
class SpectronSpectralEntropyRouter(Module)
  doc: Dynamic Spectron Router based on Spectral Entropy & Lagrangian Optimization. | Computes spectral entropy per token: $\mathcal{S}(X_t) = -\sum_d p_{t,d} \ln p_{t,d}$
class AegisSpectronFP8Attention(BaseAttention)
  doc: Aegis-Spectron-FP8 Hybrid Architecture. | Combines sliding-window local attention, true O(N) low-rank global SVD linear attention,
__init__()
__init__(in_features_or_lin1, out_features_or_lin2, rank, lora_rank, bias, device, dtype)
forward(x)
__init__(config)
forward(X)
__init__(config, hidden_size, num_heads, pretrained_attn)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class DummyConfig()
tensor_ops:
  w = torch.matmul(self.lin2.weight, self.lin1.weight)
  w = w + torch.matmul(self.lora_B, self.lora_A)
  return F.softmax(logits, dim=-1)  # [B, N, 3]
  attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)
  attn_weights = attn_weights.masked_fill(mask, mask_val)
  attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_val)
  attn = F.softmax(attn_weights, dim=-1)
  return torch.matmul(attn, V)
shapes: [1, 1, D [1, N, 1 [B, 1, N, 1 [B, N [B, N, 3 [B, N, D [B, N, h, d [w, b
parameters: torch.empty((self.out_features, self.lora_rank | torch.empty((self.lora_rank, self.in_features | torch.ones(self.all_head_size | torch.ones(self.all_head_size | torch.ones(self.all_head_size
formulas/comments:
  Token-level routing projection (removed .mean(dim=1))
  Non-negative kernel mapping phi(x) = ELU(x) + 1
  Numerator: cumsum_{j <= t} phi_K_j * V_j -> [B, N, rank, D]
  Denominator: cumsum_{j <= t} phi_K_j -> [B, N, rank]
  O(N) Recurrence using CumSum trick: y_t = gamma^t * cumsum(u_i * gamma^{-i})
features: causal,decay,delta,distill,entropy,gating,gqa,lora,peft,quant,rank,rope,rotation,svd
loc: 934