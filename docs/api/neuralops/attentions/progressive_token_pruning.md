Module neuralops.attentions.progressive_token_pruning
=====================================================
Progressive Token Pruning (PTP) – Production-Hardened (v2).

Learnable Progressive Token Pruning module with guaranteed [CLS] retention,
multi-head attention capabilities, and full multi-architecture support.

⚠️  IMPORTANT — WHAT PTP ACTUALLY DOES  ⚠️

PTP is a **compute optimization**, not a parameter compressor. It prunes
tokens at *inference time* by keeping the top-k highest-scoring tokens per
sequence (plus the always-preserved first token). The full token sequence
still flows through the model; only the attention operation is restricted
to a subset of keys/values.

Consequences:
  - Parameter count is NOT reduced (it may even slightly grow due to the
    score network).
  - KV cache is NOT reduced (all tokens are still stored).
  - FLOPs during attention ARE reduced (≈ keep_ratio × L²).
  - Long-context inference is significantly faster.

If your goal is parameter compression, use SVD/MLA/quantization instead
(or apply them *after* PTP). If your goal is inference speedup, PTP is
appropriate, but the pipeline should label it as a *compute optimizer*,
not a *compressor*.

Bug-fix map (v2):
  PTP-FIX-01  **KV-cache support** for decoder inference. Both tuple and
              DynamicCache formats are supported.
  PTP-FIX-02  **RoPE past_len offset** for KV-cache.
  PTP-FIX-03  **Removed double `scores[:, 0] = 1e4` assignment.**
  PTP-FIX-04  **Score network shares weights across layers** (optional
              shared scorer) to minimize parameter overhead.
  PTP-FIX-05  **Correct `_rotate_half` for odd head_dim.**
  PTP-FIX-06  **Position_ids forwarded to RoPE** when present.
  PTP-FIX-07  **Injection rewritten**: single-pass, no mutation during
              iteration; nested encoders handled correctly.
  PTP-FIX-08  **Removed broken `comp_config if 'comp_config' in locals()`**.
  PTP-FIX-09  **Forward sanity check** added.
  PTP-FIX-10  **Added `dense`, `c_proj`, `o_proj` aliases.**
  PTP-FIX-11  **`target_layer_idx` honored in whole-model traversal.**
  PTP-FIX-12  **Class-level `is_custom_attention = True`** (it does replace
              attention in the graph, but flags that it is not a compressor).

Functions
---------

`inject_progressive_token_pruning(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, target_layer_idx: int | None = None, shared_scorer: SharedTokenScorer | None = None, **kwargs: Any)`
:   

`inject_ptp(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, target_layer_idx: int | None = None, shared_scorer: SharedTokenScorer | None = None, **kwargs: Any)`
:   

Classes
-------

`ProgressiveTokenPruning(config: Any | None = None, d_model: int | None = None, keep_ratio: float = 0.8, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, shared_scorer: SharedTokenScorer | None = None, **kwargs: Any)`
:   Progressive Token Pruning attention.
    
    Forward:
      1. Score each token (score_network or shared scorer).
      2. Force-keep the first token ([CLS]).
      3. Keep the top-k tokens (k = ceil(L · keep_ratio)).
      4. Compute attention over the reduced set of keys/values.
      5. Return output with the original (B, L, D) shape.
    
    KV-cache: supported for decoder inference.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `gate`
    :

    `random_proj`
    :

`SharedTokenScorer(dim: int, hidden: int = 64)`
:   Optional token scorer that is shared across all PTP layers to reduce
    parameter overhead. When `share_scorer=True` is passed to all PTP
    instances, they reference the same scorer module.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.