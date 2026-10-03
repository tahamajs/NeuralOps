Module neuralops.compression_engine.admm_compressor
===================================================
ADMM Compressor: Alternating Direction Method of Multipliers for Joint Pruning and Quantization.
Registered in COMPRESSOR_REGISTRY as 'admm' and 'admm_compressor'.

Functions
---------

`admm_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for ADMM compressor.

Classes
-------

`ADMMCompressor(model: nn.Module, sparsity: float = 0.5, quant_bits: int = 4, rho1: float = 0.1, rho2: float = 0.1, max_iter: int = 8, structured_24: bool = True, block_size: tuple[int, int] = (4, 4), max_error: float = 0.45, fallback_to_original: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, quant_per_channel: bool = True, comp_config: Any = None, **kwargs: Any)`
:   ADMM-based joint structured pruning + uniform quantization engine.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor