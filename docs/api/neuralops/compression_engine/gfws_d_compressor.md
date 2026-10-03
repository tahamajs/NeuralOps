Module neuralops.compression_engine.gfws_d_compressor
=====================================================
GFWSVD: Fisher-weighted / Covariance-weighted SVD. Registered as 'gfws' and 'gfws_d'.

Functions
---------

`gfws_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   GFWS factory.

Classes
-------

`GFWSCompressor(model: nn.Module, rank: int = 64, damping: float = 0.01, max_error: float = 0.2, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, target_attention_only: bool = False, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Generalized Fisher / Covariance-Weighted SVD Compressor.
        Whitens weights using empirical input activation covariance square-root Cov(X)^{1/2},
        performs low-rank truncated SVD, and de-whitens to restore optimal compressed weights.
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error.
        - Automatic fallback to standard weight SVD if covariance whitening fails or error > max_error.
        - Safe fallback to original weights if even standard SVD exceeds max_error.
        - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
        - Unified critical-layer protection via should_compress_linear.
        - Full FP32 computation for numerical stability, casting back to original dtype only when assigning.
        - Trainable LoRA adapters (lora_A, lora_B) for parameter-efficient fine-tuning.
        - Per-layer logging and detailed compression statistics.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress layers via activation covariance whitening + low-rank SVD, with standard SVD fallback.

`GFWSLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer containing Fisher/Covariance-whitened SVD compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.gfws_d_compressor.GFWSLoRALinear`
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