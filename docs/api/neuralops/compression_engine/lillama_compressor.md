Module neuralops.compression_engine.lillama_compressor
======================================================
LiLLaMA: Local Distillation via Activation Cross-Covariance SVD.
Registered as 'lillama_class', 'lillama_compressor_class', 'lillama_factory'
and 'lillama' / 'lillama_compressor' as factory aliases.

Formulation:
  C_YX = (1/N) Y^T X                 (cross-covariance)
  C_YX ≈ U_r Σ_r V_r^T               (truncated SVD)
  W_distill = U_r Σ_r V_r^T          (distilled weight)

Functions
---------

`lillama_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for LiLLaMA compressor.

Classes
-------

`LiLLaMADistiller(model: nn.Module, tokenizer: Any | None = None, rank: int = 32, calibration_texts: list[str] | None = None, calibration_loader: Any | None = None, device: str | torch.device | None = None, max_error: float = 0.2, fallback_to_original: bool = True, skip_attention: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, max_calibration_batches: int = 8, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `category: str`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

`LiLLaMACompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 32, calibration_texts: list[str] | None = None, calibration_loader: Any | None = None, device: str | torch.device | None = None, max_error: float = 0.2, fallback_to_original: bool = True, skip_attention: bool = False, target_attention_only: bool = False, target_modules: list[str] | None = None, min_layer_dim: int = 8, skip_should_compress_filter: bool = False, max_calibration_batches: int = 8, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `category: str`
    :   The type of the None singleton.

    `is_custom_attention: bool`
    :   The type of the None singleton.