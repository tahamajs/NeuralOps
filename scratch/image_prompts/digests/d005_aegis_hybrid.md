# d005 aegis_hybrid  [attention]
aliases: aegis_hybrid,aegis_hybrid_attention,aegis_hybrid_attn
file: Codes/framework/attentions/aegis_hybrid.py  doc: Docs/content/attentions/aegis_hybrid.md
doc: Aegis-Hybrid Attention Mechanism. | Combines low-rank factorized projections, localized sliding-window dense attention, | causal/bidirectional kernelized global linear attention with token-importance pruning, | and learned gating fusion.
class AegisHybridConfig()
  doc: Configuration dataclass for AegisHybridAttention.
class FactorizedLinear(Module)
  doc: Factorized linear projection with internal rank-r base factorization and trainable residual LoRA adapter: | Base: x -> lin1(x) [in_features -> rank] -> lin2(lin1(x)) [rank -> out_features] (frozen)
class AegisHybridAttention(BaseAttention)
  doc: Aegis-Hybrid Attention combining Local Windowed Attention and Global Linear Attention. | Architectural Foundation:
__init__(in_features_or_lin1, out_features_or_lin2, rank, lora_rank, bias, device, dtype)
forward(x)
__init__(config, hidden_size, num_heads, pretrained_attn, rank, lora_rank)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class DummyConfig()
tensor_ops:
  w = torch.matmul(self.lin2.weight, self.lin1.weight)
  w = w + torch.matmul(self.lora_B, self.lora_A)
  Q = self.q_proj(hidden_states).view(B, N, self.num_heads, Dh).transpose(1, 2)  # [B, H, N, Dh]
  K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2)  # [B, H_kv, N_k, Dh]
  V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2)  # [B, H_kv, N_k, Dh]
  head_mask_tensor = head_mask.view(1, -1, 1, 1)
  scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N_k]
  scores = scores.masked_fill(~band_mask.unsqueeze(0).unsqueeze(0), -10000.0)
shapes: [B, 1, N, 1 [B, H, Dh [B, H, Dh, Dh [B, H, N, 1 [B, H, N, Dh [B, H, N, Dh, Dh [B, N, H [B, N, H, 1
parameters: torch.empty((self.out_features, self.lora_rank | torch.empty((self.lora_rank, self.in_features
formulas/comments:
  In causal masks [B, 1, N, N_k], row i only attends to j <= i.
  ============== Branch 1: Local Window Attention ==============
  2. Hybrid Combination: Local Window Attention + True O(N) Global Linear Attention
  TRUE O(N) CAUSAL LINEAR ATTENTION using Cumulative Sum
  TRUE O(N) BIDIRECTIONAL LINEAR ATTENTION
features: adaptive,causal,delta,distill,gating,gqa,lora,peft,rank,rope,rotation,svd,window
loc: 939