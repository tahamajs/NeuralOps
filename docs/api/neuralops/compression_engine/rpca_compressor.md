Module neuralops.compression_engine.rpca_compressor
===================================================
Robust PCA: decompose W = L + S, factorizing low-rank component L with structural SVDLinear.
Registered in COMPRESSOR_REGISTRY as 'rpca', 'rpca_factory', and 'rpca_compressor_factory'.

Functions
---------

`rpca_compressor_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   RPCA factory function returning (compressed_model, stats) when requested.

`rpca_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   RPCA factory function returning (compressed_model, stats) when requested.

Classes
-------

`RPCACompressor(model: nn.Module, rank: int = 64, mu: float = 0.001, max_iter: int = 5, min_rank: int = 8, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, target_modules: list[str] | None = None, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Robust Principal Component Analysis (RPCA) Low-Rank Compressor.
        
        Decomposes weight matrices W into low-rank component L and sparse outlier matrix S
        via Principal Component Pursuit / alternating soft-thresholding:
            min ||L||_* + lambda ||S||_1  s.t.  W = L + S
        Factorizes the low-rank component L into symmetric factors (U * sqrt(Sigma), sqrt(Sigma) * V^T)
        and replaces nn.Linear or Conv1D with factorized SVDLinear modules with trainable LoRA fine-tuning adapters.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear layers in the model using RPCA decomposition.

    `compress_linear(self, layer: nn.Module, rank: int) ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear layer via RPCA and return factorized SVDLinear with LoRA.