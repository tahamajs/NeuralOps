Module neuralops.compression_engine.flap
========================================
FLAP: Fluctuation-Based Adaptive Structured Pruning (Corrected).

Registered in COMPRESSOR_REGISTRY as 'flap' and 'flap_compressor'.
Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with reconstruction validation,
Conv1D support, and safe fallback.

Functions
---------

`flap_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for FLAP compressor.

Classes
-------

`FLAPCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.2, metric_type: str = 'WIFV', max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, structured_24: bool = False, target_modules: list[str] | None = None, cfg: Any | None = None, num_calibration_batches: int = 8, **kwargs: Any)`
:   FLAP Compressor: Fluctuation-based Adaptive Structured Pruning.
    
    Production hardening features:
    - Robust LoRALinear and Conv1D support (transpose + adapter preservation).
    - target_attention_only restricts pruning to attention projections.
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if error exceeds threshold.
    - Real metrics (effective_parameter_ratio, num_compressed, num_fallback, sparsity).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor