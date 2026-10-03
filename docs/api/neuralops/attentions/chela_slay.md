Module neuralops.attentions.chela_slay
======================================
CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel — Production-Hardened (v3).

Fixes (v3) relative to v2:
  CS-FIX-16  `_safe_int`, `_safe_float` helpers handle `None` values and missing
             attributes uniformly. Previously `int(None)` could crash __init__.
  CS-FIX-17  `_extract_attn_config` robustly resolves hidden_size, num_heads,
             head_dim, num_kv_heads across HF configs (dict or object) and
             uses pretrained attention as a fallback source.
  CS-FIX-18  `_copy_pretrained` never raises on missing weights; instead it
             logs and falls back to initialized weights (so injection still
             succeeds on unexpected architectures).
  CS-FIX-19  `inject_chela_slay` calls `.to(device, dtype)` after construction
             so the new module lands on the correct device.
  CS-FIX-20  `_copy_pretrained` handles the case where weights come from
             HF's `base_layer` (PEFT-wrapped) transparently.
  CS-FIX-21  `_copy_pretrained` never resets weights to zero when a source
             tensor shape cannot be reconciled; it leaves the init intact.
  CS-FIX-22  `comp_config=None` is tolerated everywhere.
  CS-FIX-23  `key`/`value` aliases point to `W_k`/`W_v` (not `k_up`/`v_up`
             which do not exist here).
  CS-FIX-24  `gate_init_bias` no longer defaulted via `kwargs` — it uses the
             constructor argument directly.
  CS-FIX-25  `_causal_linear_attention` uses `denom.abs() + eps` consistently
             (no sign flips).

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_chela_slay(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   

Classes
-------

`ChelaSlayAttention(config: Any | None = None, rank: int = 64, feature_dim: int = 128, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, gate_init_bias: float = 5.0, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `gate`
    :

    `k_lin`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `query`
    :

    `random_proj`
    :

    `v_lin`
    :

    `v_proj`
    :

    `value`
    :

    ### Methods

    `storage_bytes(self) ‑> dict[str, int]`
    :