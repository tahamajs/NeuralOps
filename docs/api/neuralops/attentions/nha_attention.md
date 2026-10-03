Module neuralops.attentions.nha_attention
=========================================
Native Hybrid Attention (NHA) – Production-Hardened (v2).

Fuses localized sliding-window attention with global linear Random Fourier
Features (RFF) via dynamic token-level gating and exact pretrained weight
transfer.

Fixes (v2):
  NHA-FIX-01  **Registry keys separated**: class under `nha_class` /
              `nha_attention_class`; injector under `nha` / `inject_nha`.
  NHA-FIX-02  **Removed `return None` in forward** — RoPE failures now
              propagate as RuntimeError, not silent crash.
  NHA-FIX-03  **Removed all `print()`** — uses `logger`.
  NHA-FIX-04  **`_rotate_half` handles odd head_dim.**
  NHA-FIX-05  **KV-cache support** (tuple + HF DynamicCache).
  NHA-FIX-06  **`num_kv_heads` clamped** to `num_heads`.
  NHA-FIX-07  **`is_nested_encoder` initialized to False.**
  NHA-FIX-08  **Cross-attention raises NotImplementedError.**
  NHA-FIX-09  **No silent padding/truncation** of output dim.
  NHA-FIX-10  **Idempotency guard in inject_nha.**
  NHA-FIX-11  **`self.rotary_emb` receives a proper `x` reference** (q or k).
  NHA-FIX-12  **`comp_config` resolved safely** (not via `or`).
  NHA-FIX-13  **Forward sanity check** after weight transfer.
  NHA-FIX-14  **LoRA dtype-safe forward.**
  NHA-FIX-15  **`output_scale` applied consistently** in both nested and
              non-nested paths.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_native_hybrid_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Injection hook for NativeHybridAttention.

`inject_nha(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Injection hook for NativeHybridAttention.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`NativeHybridAttention(config: Any | None = None, rank: int = 64, window_size: int = 64, rff_features: int = 128, dropout: float = 0.1, pretrained_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Native Hybrid Attention: local sliding-window + global RFF, fused by a
    token-level learned gate.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.