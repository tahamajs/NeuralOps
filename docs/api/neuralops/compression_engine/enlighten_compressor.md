Module neuralops.compression_engine.enlighten_compressor
========================================================
ENLighten: Low-Rank SVD + 2:4 Structured Sparsity (Photonic-Aligned).
Registered as 'enlighten' and 'enlighten_compressor'. Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with vectorization, error validation, and safe fallback.

Functions
---------

`enlighten_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Enlighten compressor factory function.

`vectorized_24_prune(residual: torch.Tensor) ‑> torch.Tensor`
:   Vectorized 2:4 structured sparsity along row blocks (groups of 4 input columns).
    Handles arbitrary in_features by zero-padding to a multiple of 4.

Classes
-------

`ENLightenCompressor(model: nn.Module, rank: int = 64, sparsity: float = 0.5, structured_24: bool = True, max_error: float = 0.45, target_attention_only: bool = False, fallback_to_original: bool = True, max_retries: int = 2, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   ENLighten Compressor: Low-Rank SVD + 2:4 Structured Sparsity.
        Decomposes weights into low-rank factors plus 2:4 structured sparse residuals,
        preserving Frobenius energy while reducing memory bandwidth and arithmetic complexity.
        
        Production hardening features:
        - Robust LoRALinear wrapper and Conv1D support (with transpose and adapter preservation).
        - target_attention_only parameter: restricts compression to attention projections and freezes non-target layers.
        - Relative Frobenius reconstruction error validation with configurable max_error.
        - Automatic rank escalation retries when error threshold is exceeded.
        - Safe fallback to original module if error remains high or decomposition is unstable, with fallback freezing.
        - Fully vectorized 2:4 structured pruning eliminating Python loop overhead.
        - Critical-layer protection via should_compress_linear.
        - Real compression metrics recording (effective_parameter_ratio, num_compressed, num_fallback, etc.).

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear and Conv1D layers using low-rank SVD + 2:4 structured sparse residuals.