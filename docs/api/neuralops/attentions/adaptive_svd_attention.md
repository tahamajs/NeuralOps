Module neuralops.attentions.adaptive_svd_attention
==================================================
Adaptive SVD Attention — Production-Hardened (v2).

Low-rank factorized attention with dynamic cumulative energy-thresholded SVD
and trainable residual adapters:

    W ≈ U_r V_r + W_lora_B · W_lora_A
    where  U_r = U_{:, :r} · √Σ_r,  V_r = √Σ_r · V^T_{:r, :}
    and    r = argmin_{r} [ Σ_{i=1}^r σ_i² / Σ_j σ_j² ≥ τ ]

Fixes (v2):
  ASVD-FIX-01  **`--rank` now reaches the layer.** Previous priority order
               placed `comp_config.svd_rank` above `kwargs["rank"]`, silently
               freezing the rank at a config default regardless of CLI value.
  ASVD-FIX-02  **Energy threshold default is 0.999.** With 0.92 the effective
               rank was always the same (~40) for typical transformer weights,
               making rank sweeps meaningless. 0.999 lets `svd_rank` actually
               matter.
  ASVD-FIX-03  **`AdaptiveSVDLinear.is_custom_attention = False`** and new
               flag `is_svd_linear = True` so framework counters classify it
               as a compressed linear layer, not as an attention module.
  ASVD-FIX-04  **Single source of truth for head_dim/num_heads resolution**
               (config → pretrained → fallback), no conflicting reads.
  ASVD-FIX-05  **`_rotate_half` moved to module scope** (no per-forward
               closure allocation).
  ASVD-FIX-06  **`forward` always returns `(output, attn_weights)`.** The
               T5-style `position_bias` is no longer returned in the second
               slot (that broke `out, attn = attn(...)` unpacking).
  ASVD-FIX-07  **Idempotency guard** in injector.
  ASVD-FIX-08  **`is_causal` computed once in `__init__`.**
  ASVD-FIX-09  **dtype-safe LoRA forward.**
  ASVD-FIX-10  **Bias frozen after `copy_weights`.**
  ASVD-FIX-11  **Storage accounting** (`_orig_storage_bytes` /
               `_comp_storage_bytes`).
  ASVD-FIX-12  **`config` attribute always set** so `self.config` never fails.
  ASVD-FIX-13  **`inject_adaptive_svd` respects `kwargs["rank"]` first.**

Functions
---------

`inject_adaptive_svd(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`AdaptiveSVDAttention(config: Any, svd_rank: int | None = None, residual_rank: int = 16, energy_threshold: float = 0.999, pretrained_attn: torch.nn.modules.module.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Multi-head attention with adaptive SVD + residual factorized projections.
    
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin_alias`
    :

    `q_lin_alias`
    :

    `v_lin_alias`
    :

    ### Methods

    `compute_bias(self, query_length, key_length, device=None)`
    :

    `storage_bytes(self) ‑> dict`
    :

`AdaptiveSVDLinear(in_features: int, out_features: int, svd_rank: int, residual_rank: int = 16, bias: bool = True, trainable_svd: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Frozen SVD base + trainable LoRA residual.
    
    Forward:
        y = x V^T U^T + x lora_A^T lora_B^T + b
    
    Storage:
        U (out, r), V (r, in), lora_A (residual_rank, in), lora_B (out, residual_rank)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_compressed_svd: bool`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `is_svd_linear: bool`
    :   The type of the None singleton.

    ### Instance variables

    `A: torch.nn.parameter.Parameter | None`
    :

    `B: torch.nn.parameter.Parameter | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `reset_parameters(self) ‑> None`
    :

    `set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) ‑> None`
    :