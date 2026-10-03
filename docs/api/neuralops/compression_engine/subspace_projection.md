Module neuralops.compression_engine.subspace_projection
=======================================================
Subspace Projection Compressor: Low-Rank Subspace Projection and Factorization.

Registered in COMPRESSOR_REGISTRY as 'subspace_projection',
'subspace_projection_compressor'.

Bug-fix map (v2):
  SP-FIX-01  is_custom_attention = False on SubspaceLinear and compressor.
  SP-FIX-02  **Added missing `import logging` and `logger`.**
  SP-FIX-03  **dtype-safe forward**: A, B, bias are cast to x_flat's dtype.
  SP-FIX-04  **LoRA transfer handles Parameter and Module** (uses
             `getattr(m, "weight", m)`).
  SP-FIX-05  **`target_layers=[]` treated as None.**
  SP-FIX-06  **`method` documented as currently only 'svd' is implemented.**
  SP-FIX-07  **`param_cap` formula corrected** (no spurious `-1`).
  SP-FIX-08  **Forward sanity check** added at end of `compress()`.
  SP-FIX-09  **Honest storage accounting** (`storage_compression_ratio`).
  SP-FIX-10  **LoRA wrapper preserved** when `module.base_layer` exists.
  SP-FIX-11  **Bias copy under `torch.no_grad()`.**
  SP-FIX-12  **Skip already-compressed SubspaceLinear modules**
             (idempotency).

Functions
---------

`subspace_projection_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   

Classes
-------

`SubspaceLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Subspace Projection Factorized Linear Layer with optional LoRA adapter.
    
    W ≈ B @ A, where:
      A (rank, in_features):  orthonormal / projection basis
      B (out_features, rank): coefficient expansion matrix
    
    Forward:
      y = ((x @ A^T) @ B^T) + bias + (lora_B @ lora_A @ x) * scaling
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, rank: int = 64, method: str = 'svd', use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.subspace_projection.SubspaceLinear`
    :   Project linear or Conv1D weight matrix onto top-rank subspace.

    ### Instance variables

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

    `get_weight(self) ‑> torch.Tensor`
    :

`SubspaceProjectionCompressor(model: nn.Module | None = None, rank: int = 64, method: str = 'svd', cfg: Any | None = None, target_layers: list[str] | None = None, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.5, fallback_to_original: bool = True, **kwargs: Any)`
:   Subspace Projection Compression Engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear: nn.Module) ‑> neuralops.compression_engine.subspace_projection.SubspaceLinear`
    :