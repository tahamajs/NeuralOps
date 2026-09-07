# d008 aegis_v2  [attention]
aliases: aegis_v2
file: Codes/framework/attentions/aegis_v2.py  doc: Docs/content/attentions/aegis_v2.md
doc: Aegis-Attention V2 Module. | Implements hybrid low-rank factorized attention with local sliding window attention, | global linear kernel attention with adaptive soft-pruning gating, and dynamic multi-head fusion. | Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
class AegisAttentionConfigV2()
  doc: Configuration dataclass for AegisAttentionV2.
class FactorisedLinear(Module)
  doc: SVD-factorized linear layer with energy-based rank truncation and trainable LoRA adapters. | Base SVD weights are frozen; LoRA branch provides trainable parameters.
class AegisAttentionV2(BaseAttention)
  doc: AEGIS-Attention V2 Multi-Head Layer.
__init__(in_features, out_features, rank, bias, lora_rank, lora_alpha, energy)
forward(x)
__init__(config, pretrained_attn, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
class DummyConfig()
tensor_ops:
  W = W.view(self.out_features, rep, -1).mean(dim=1)
  bias = bias.view(self.out_features, rep).mean(dim=1)
  base_w = torch.matmul(self.up.weight, self.down.weight)
  Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
  K = self.k_proj(kv_input).view(B, N_kv, self.num_kv_heads, self.head_dim)
  V = self.v_proj(kv_input).view(B, N_kv, self.num_kv_heads, self.head_dim)
  Q_r, K_r = apply_rotary_pos_emb(Q.transpose(1, 2), K.transpose(1, 2), cos, sin)
  Q, K = Q_r.transpose(1, 2), K_r.transpose(1, 2)
shapes: [w, b
parameters: torch.randn(lora_rank, in_features | torch.zeros(out_features, lora_rank
features: adaptive,causal,distill,gating,gqa,lora,rank,rope,rotation,svd,window
loc: 699