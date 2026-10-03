Module neuralops.compression_engine.swift_svd
=============================================
Swift-SVD: fast closed-form truncated SVD with symmetric square-root singular value distribution.
Registered as 'swift_svd'.

Functions
---------

`inject_swift_svd(model: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook for Swift-SVD.

`swift_svd_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Swift-SVD factory function.

Classes
-------

`SwiftSVDCompressor(model: nn.Module, rank: int | None = None, rank_ratio: float = 0.5, target_modules: list[str] | None = None, max_error: float = 0.6, max_retries: int = 3, fallback_to_original: bool = True, validation_calibration: bool = False, calibration_threshold: float = 0.1, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Swift-SVD Compressor: Fast Closed-Form Truncated SVD.
        
        Decomposes dense weight matrices into rank-r factorized layers with symmetric
        square-root singular value splitting (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T)
        and replaces nn.Linear/Conv1D with factorized SVDLinear modules.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all eligible linear and Conv1D layers.

    `compress_linear(self, layer: nn.Module, rank: int | None = None, name: str = '') ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :   Decompose a single linear or Conv1D layer via truncated SVD.
        Returns (svd_layer_or_None, relative_frobenius_error, final_rank).