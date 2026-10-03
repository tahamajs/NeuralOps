Module neuralops.compression_engine.direct_svd
==============================================
Direct SVD Compressor: Truncated SVD low-rank factorized layer replacement.
Registered as 'direct_svd'. Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support, rank bounds, and safe fallback.

Functions
---------

`direct_svd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Direct SVD factory function.

`direct_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Direct SVD factory function.

Classes
-------

`DirectSVDCompressor(model: nn.Module, rank: int | None = 32, min_rank: int = 16, rank_ratio: float | None = None, max_error: float = 0.35, fallback_to_original: bool = True, protect_critical: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, target_attention_only: bool = False, max_retries: int = 2, target_modules: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Direct SVD Compressor.
        
        Performs direct truncated SVD on target linear and Conv1D layers:
            W ≈ U_r * Sigma_r * V_r^T
        Allocates singular energy symmetrically:
            U_factor = U_r * sqrt(Sigma_r)
            V_factor = sqrt(Sigma_r) * V_r^T
        and replaces dense layers with factorized SVDLinear modules.
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error.
        - Automatic rank escalation retries when error threshold is exceeded.
        - Safe fallback to original layer if error remains high or decomposition fails.
        - Fallback and non-target layers frozen (requires_grad=False) during recovery.
        - Robust LoRALinear and Conv1D support (with transpose and adapter preservation).
        - target_attention_only parameter: restricts SVD factorization to attention projections.
        - Strict rank safeguard: enforces r < min(in_f, out_f) and parameter reduction r * (in + out) < in * out.
        - Unified critical-layer protection via should_compress_linear.
        - Full FP32 computation for numerical stability.
        - Real metrics recording: original_layer_count, compressed_layer_count, fallback_layer_count, effective_parameter_ratio.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target linear and Conv1D layers across the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :   Decompose a single linear or Conv1D layer via SVD with symmetric factor distribution.
        Returns: (svd_layer, relative_reconstruction_error, chosen_rank)