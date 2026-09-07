# d093 nexus  [attention]
aliases: nexus
file: Codes/framework/attentions/nexus.py  doc: Docs/content/attentions/nexus.md
doc: NEXUS: Neural Efficient X-scoring Unified Sparse Attention. | ============================================================= | Hybrid sparse attention architecture combining multi-scale sparsity mechanisms: | 1. XAttention antidiagonal block scoring (ICML 2025) - training-free block routing
class LoRALinear(Module)
class NEXUSConfig()
  doc: Configuration dataclass for NEXUS Attention parameters. | Attributes:
class XAttentionScorer()
  doc: Training-free block importance scoring via antidiagonal sub-sampled correlation. | Subsamples query and key matrices at regular stride $\sigma$ and evaluates coarse
class SeerGating(Module)
  doc: Learnable block-level sparsity gating module (SeerAttention). | Uses a low-rank query-key pooling network to predict head-wise dynamic sparsity budgets
class DualStageStripeAttention(Module)
  doc: Dual-stage sparse attention engine with stripe granularity. | Implements intra-block dense attention combined with top-k inter-block attention
class SpectraformerRectify(Module)
  doc: Global linear attention module using Random Fourier Features (RFF). | rff_features: Feature mapping dimensionality $F$.
class NEXUSAttention(BaseAttention)
  doc: NEXUS: Neural Efficient X-scoring Unified Sparse Attention. | Unified multi-scale sparse attention with training-free block scoring, learnable gating,
__init__(in_features, out_features, bias, lora_rank, lora_alpha, freeze_base, device, dtype)
forward(x)
__init__(head_dim, num_heads, gate_rank, top_k, temperature)
forward(Q, K, n_blocks, training)
__init__(cfg, head_dim)
forward(Q, K, V, token_valid, seer_mask)
__init__(head_dim, rff_features)
forward(Q, K, V)
__init__(config, model_config, nexus_cfg, layer_idx, total_layers, pretrained_attn, bert_mode, lora_rank, lora_alpha, is_decoder)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
tensor_ops:
  attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * scale
  q_gate = self.q_pool(Q_pooled).view(B, H, 1, gr)
  k_gate = self.k_pool(K_pooled).view(B, H, gr, 1)
  torch.matmul(q_gate, k_gate).squeeze(-1).squeeze(-1) / self.temperature
  scores = torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale
  scores = scores.masked_fill(~stripe_mask, -10000.0)
  scores = scores.masked_fill(~intra_valid.bool(), -10000.0)
  else F.softmax(scores, dim=-1)
shapes: [B, H [B, H, D [B, N, H * D [w, b
parameters: torch.empty((out_features, in_features | torch.empty(out_features, **factory_kwargs | torch.empty((lora_rank, in_features | torch.empty((out_features, lora_rank | proj_init / math.sqrt(head_dim
formulas/comments:
  ============================================================================
  ============================================================================
  ============================================================================
  ============================================================================
  ============================================================================
  ============================================================================
  ============================================================================
  ============================================================================
  meaning blocks in sentence 2 (indices 2–3 for L=128/block_size=32)
  ============================================================================
features: adaptive,causal,distill,gating,gqa,lora,pool,rank,rope,rotation,sparsity,stride,topk
loc: 1000