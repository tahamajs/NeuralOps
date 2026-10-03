Module neuralops.compression_engine.frandi_compressor
=====================================================
FRanDI: Feature-Regression Aligned SVD Compressor — Production-Hardened (v2).

Registered as 'frandi' and 'frandi_compressor'.

Functions
---------

`frandi_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for FRanDI compressor.

Classes
-------

`FRanDICompressor(model: nn.Module | None = None, rank: int = 8, energy_threshold: float | None = None, calibration_loader: Any | None = None, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.5, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, num_calibration_batches: int = 16, **kwargs: Any)`
:   FRanDI (Feature-Regression aligned SVD) Compressor.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`FRanDILinear(in_features: int, out_features: int, rank: int = 8, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Feature-Regression Aligned Factorized Linear Layer with optional LoRA.
    
    W ≈ B @ A
      - A (rank, in_features): input feature-aligned basis
      - B (out_features, rank): output expansion matrix
    
    Forward:
      y = (x @ A^T) @ B^T + bias
          + (alpha/r) * (x @ lora_A^T @ lora_B^T)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 8, energy_threshold: float | None = None, cov_matrix: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.frandi_compressor.FRanDILinear`
    :

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

    `merge(self) ‑> None`
    :   Fold LoRA delta into base factors (approximate, additive).

    `merge_and_unload(self) ‑> neuralops.compression_engine.frandi_compressor.FRanDILinear`
    :

    `unmerge(self) ‑> None`
    :