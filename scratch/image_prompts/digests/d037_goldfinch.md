# d037 goldfinch  [attention]
aliases: goldfinch,goldfinch_ahn
file: Codes/framework/attentions/goldfinch_ahn.py  doc: Docs/content/attentions/goldfinch_ahn.md
doc: GoldFinch-AHN: Adaptive Hybrid Network Combining Sliding Window and Gated DeltaNet Memory (Hardened). | Combines short-term multi-head sliding-window attention with long-term GatedDeltaNet recurrent memory | via a learned input-dependent dynamic gating mechanism initialized for stable representation transfer. | Supports:
class LoRALinear(Module)
class GatedDeltaNetLayer(Module)
  doc: Gated DeltaNet Recurrent Memory Layer.
class GoldFinchAHNBlock(BaseAttention)
  doc: GoldFinch-AHN: Adaptive Hybrid Network Attention Block. | Combines multi-head sliding-window attention with long-term GatedDeltaNet memory
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(d_model, use_spectral_norm)
forward(x)
__init__(config, rank, d_model, num_heads, window_size, is_causal, pretrained_attn, old_attn, gate_init_bias, comp_config)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  w = w.reshape(w.size(0), -1)
  x = x.contiguous()
  q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
  k = self.W_k(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  v = self.W_v(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
  pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
  scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
  scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), mask_val)
shapes: [B, L, 1 [B, L, D
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | torch.tensor(-3.0
formulas/comments:
  Preservation gate: sigmoid(-3) ≈ 0.047 → ~95% passthrough at init.
features: adaptive,causal,delta,distill,gating,gqa,lora,peft,rank,rope,rotation,window
loc: 666