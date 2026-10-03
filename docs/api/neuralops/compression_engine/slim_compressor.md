Module neuralops.compression_engine.slim_compressor
===================================================
SLiM: Structured Low-Rank Integrated Matrix Compression (Quantization + 2:4 Sparsity + SVD Residual Correction).
Registered in COMPRESSOR_REGISTRY as 'slim' and 'slim_compressor'.

Mathematical Formulation:
1. Uniform/Non-Uniform b-bit Quantization:
   $$W_q = \operatorname{Quant}_b(W_0)$$
2. 2:4 Semi-Structured Sparsity:
   $$W_{sp} = \Pi_{2:4}(W_q, |W_q|)$$
3. Pruning & Quantization Residual:
   $$R = W_0 - W_{sp}$$
4. Low-Rank Singular Value Decomposition Correction:
   $$R \approx U_r \Sigma_r V_r^\top = L_r$$
5. Final Reconstructed Weight:
   $$W_{slim} = W_{sp} + L_r$$

Functions
---------

`slim_compressor_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> Any`
:   Factory function for SLiM compressor.

`slim_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> Any`
:   Factory function for SLiM compressor.

Classes
-------

`SLiMCompressor(model: nn.Module, bits: int = 4, sparsity: float = 0.5, rank: int = 8, structured_24: bool = True, target_attention_only: bool = False, max_error: float = 0.5, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SLiM Compressor: One-shot joint b-bit quantization, 2:4 structured sparsity, and low-rank residual recovery.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Executes SLiM compression pipeline across all Linear and Conv1D layers.