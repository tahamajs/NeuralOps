Module neuralops.attentions.fusion_attn
=======================================
FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module (Production-Hardened v2).

Hybrid sparse attention with:
  - Short-sequence fast path (full attention via SDPA / manual)
  - Long-sequence block-gated sparse attention (MoBA-style)
  - Per-head dynamic gating for block selection
  - RoPE support with partial-dim slicing
  - GQA with repeat_kv
  - Nested encoder bypass
  - LoRA adapters on frozen base projections
  - Strict causal masking across both paths
  - Padding masking across both paths
  - KV-cache support

Bug-fix map (v2):
  FA-FIX-01  `logger` defined at module scope.
  FA-FIX-02  Fast-path SDPA now honors attention_mask / head_mask /
             output_attentions; falls back to manual path when needed.
  FA-FIX-03  Block-path causal mask uses ABSOLUTE positions, not a
             fixed local diagonal.
  FA-FIX-04  Block-path padding mask is gathered with the same indices
             as K/V, so masking is aligned.
  FA-FIX-05  `top_indices[..., 0] = 0` replaced with proper
             "force-include block 0" via top-(k-1) + cat.
  FA-FIX-06  `past_key_value` is read and updated for self-attention.
  FA-FIX-07  RoPE fallback uses `hidden_states` (was `v`).
  FA-FIX-08  NaN/Inf guards on block-path output.
  FA-FIX-09  `head_mask` applied in block path.
  FA-FIX-10  `output_attentions` returns block-attention weights.
  FA-FIX-11  `inject_fusion_attn` logs exceptions with traceback.
  FA-FIX-12  Idempotency guard in injector.
  FA-FIX-13  `num_kv_heads` reads `num_kv_heads` fallback (was only
             `num_key_value_heads`).
  FA-FIX-14  Debug `print` removed.
  FA-FIX-15  Per-head gating MLP (no more shared gate).
  FA-FIX-16  Gate uses per-token centroid scoring (mean over block is
             kept, but head_id is embedded in the gate input).
  FA-FIX-17  Block loop replaced with vectorized gather where possible.
  FA-FIX-18  Causal masking applied to gate_scores before top-k in
             decoders, so top-k only considers visible blocks.

Functions
---------

`inject_fusion_attn(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`FUSIONATTN(config: Any | None = None, hidden_size: int = 768, num_heads: int = 12, block_size: int = 64, top_k_blocks: int = 4, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Hybrid sparse attention with block gating.
    
    Two execution paths:
      * Short sequence (N <= 128) or cross-attention  → dense attention
      * Long sequence                                  → block-gated sparse
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k`
    :

    `W_q`
    :

    `W_v`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `query`
    :

    `v_lin`
    :

    `value`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :