Module neuralops.attentions.hgrn2_outer
=======================================
HGRN-2 Outer-Product Recurrent Linear Attention — Production-Hardened v2.

Multi-head bidirectional/causal outer-product associative matrix memory
updates with log-space bounded decay gating and denominator normalization.

Fixes (v2):
  HGRN2-FIX-01  **Chunked cumsum scan** replaces the naive Python loop.
                Complexity drops from O(L) sequential steps to O(L/C)
                chunks. For L=128, C=64 → 2 chunks vs 128 steps.
  HGRN2-FIX-02  **Denominator z now decays with the same gate as S.**
                Previously `z = z + k_t` (no decay) caused the output
                to shrink toward zero as sequence length grew.
  HGRN2-FIX-03  **Idempotency guard** in `inject_hgrn2_outer`.
  HGRN2-FIX-04  **`LoRALinear` dtype/device safe**.
  HGRN2-FIX-05  **Injector returns the injected module** (not the layer)
                so caller counters increment.
  HGRN2-FIX-06  **Injector detects decoder** and passes `is_decoder`.
  HGRN2-FIX-07  **`is_custom_attention = False`** on `LoRALinear`.
  HGRN2-FIX-08  **Storage accounting** in `storage_bytes()`.
  HGRN2-FIX-09  **`state_norm` applied on head_dim axis** consistently.
  HGRN2-FIX-10  **Robust mask handling** for 2D/4D masks.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_hgrn2_outer(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`HGRN2OuterBlock(config: Any | None = None, rank: int | None = None, dim: int | Any | None = None, num_heads: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, chunk_size: int = 64, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :