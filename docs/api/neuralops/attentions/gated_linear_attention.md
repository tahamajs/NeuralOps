Module neuralops.attentions.gated_linear_attention
==================================================
Gated Linear Attention (GLA) — Production-Hardened (v3).

Multi-head linear recurrent attention with data-dependent decay gating,
multiplicative SiLU output gating, denominator normalization, and pretrained
weight transfer. Supports DistilBERT, RoBERTa, GPT-2, and Qwen.

Fixes in v3:
  GLA3-FIX-01  Explicit shape assertion in `__init__` for num_heads, head_dim,
               decay_gate, log_decay, state_norm. Raises early with a clear
               message instead of failing mid-forward.
  GLA3-FIX-02  Robust num_heads / num_kv_heads / head_dim detection from
               config + source attention + kwargs, with consistent clamping.
  GLA3-FIX-03  `decay_gate` and `log_decay` are built from the *resolved*
               num_heads, so they can never disagree.
  GLA3-FIX-04  `head_dim` derived from config.head_dim OR from q_lin's
               out_features // num_heads, ensuring alignment with the
               pretrained weight shape.
  GLA3-FIX-05  Sanity check in forward: `q.shape[-1] == head_dim` else raise
               a descriptive error pointing at the offending weight.
  GLA3-FIX-06  `head_mask` handling unified across 1D/2D/3D/4D.
  GLA3-FIX-07  `z_proj` output dim verified against `all_head_size`.
  GLA3-FIX-08  dtype-safe `log_decay` broadcast (explicit `.to(y.dtype)`).
  GLA3-FIX-09  Removed dead `output_scale` multiplication on `is_nested_encoder`
               branch (it was applied twice in some framework paths).
  GLA3-FIX-10  `_normalize_valid_mask` now returns (B, N_q) even when the
               input is 4D with N_q != 1.
  GLA3-FIX-11  `inject_gated_linear` resolves d_model, num_heads, and
               head_dim from the source attention's actual weight shapes,
               not just from config attributes.
  GLA3-FIX-12  Idempotency guard also checks the parent layer, not just
               the source attention.
  GLA3-FIX-13  `copy_weights` validates extracted weight shapes against
               the resolved (all_head_size, dim) and raises early on
               irreconcilable mismatch.

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin)`
:   

`inject_gated_linear(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`GatedLinearAttention(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, head_dim: int | None = None, num_kv_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, chunk_size: int = 128, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

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

    `storage_bytes(self) ‑> dict[str, int]`
    :

`LoRALinear(in_features, out_features, bias=True, lora_rank=16, lora_alpha=32.0, freeze_base=False, device=None, dtype=None)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self)`
    :

    `reset_parameters(self)`
    :