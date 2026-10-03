Module neuralops.compression_engine.saes_svd_v2
===============================================
SAES-SVD v2: Spectrum-Aware Error-Suppressed Singular Value Decomposition.
Registered in COMPRESSOR_REGISTRY as 'saes_svd_v2' and 'saes'.

Mathematical Formulation:
1. Truncated SVD Baseline:
   $$W_0 \approx U_r \Sigma_r V_r^\top = A_0 B_0$$
2. Spectral Energy Distribution:
   $$\rho_k = \frac{\sigma_k^2}{\sum_{j=1}^r \sigma_j^2}$$
3. Error-Suppression Iterations:
   $$E_t = W_0 - A_t B_t$$
   $$A_{t+1} = A_t + E_t B_t^\top (B_t B_t^\top + \epsilon I)^{-1}$$
   $$B_{t+1} = B_t + (A_{t+1}^\top A_{t+1} + \epsilon I)^{-1} A_{t+1}^\top E_t$$

Functions
---------

`saes_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   SAES-SVD v2 factory function.

`saes_svd_compressor_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   SAES-SVD v2 factory function.

`saes_svd_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   SAES-SVD v2 factory function.

`saes_svd_v2_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   SAES-SVD v2 factory function.

Classes
-------

`SAESSVDCompressorV2(model: nn.Module, rank: int | None = None, rank_ratio: float = 0.5, correcting_steps: int = 2, target_modules: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.6, fallback_to_original: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SAES-SVD v2 Compressor with iterative spectrum-aware residual correction,
        Frobenius validation, and safe fallback.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear and Conv1D layers in the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None, name: str = '') ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear or Conv1D layer with spectrum-aware error suppression.

`SAESCompressor(model: nn.Module, rank: int | None = None, rank_ratio: float = 0.5, correcting_steps: int = 2, target_modules: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.6, fallback_to_original: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SAES-SVD v2 Compressor with iterative spectrum-aware residual correction,
        Frobenius validation, and safe fallback.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress all linear and Conv1D layers in the model.

    `compress_linear(self, layer: nn.Module, rank: int | None = None, name: str = '') ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :   Decompose a single linear or Conv1D layer with spectrum-aware error suppression.