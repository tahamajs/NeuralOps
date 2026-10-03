Module neuralops.compression_engine.adaptive_ranks
==================================================
Adaptive Ranks SVD Compressor: dynamic rank selection per layer based on singular value energy spectra.

Dynamic cumulative singular value energy thresholding:
    $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
with minimum rank floor:
    $$r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor)$$
activation covariance whitening:
    $$M = W \cdot \operatorname{Cov}(X)^{1/2}$$
and symmetrical $\sqrt{\Sigma}$ singular value distribution:
    $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
Registered canonically as 'adaptive_ranks' and 'adaptive_svd'.

Functions
---------

`adaptive_ranks_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Adaptive ranks factory function with robust None/dict/object config handling.

`adaptive_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Adaptive ranks factory function with robust None/dict/object config handling.

Classes
-------

`AdaptiveRanksCompressor(model: nn.Module, energy_threshold: float = 0.92, min_rank: int = 16, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, use_jg_svd: bool = False, use_cholesky_jg: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Adaptive Ranks SVD Compressor.
        
        Dynamically determines ranks per layer according to cumulative singular value energy threshold:
            $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        with minimum rank floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor, \lfloor 0.10 \cdot d_{out} \rfloor)$,
        activation covariance whitening, and symmetrically distributed singular values:
            $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
        - Safe fallback to original weights by default (fallback_to_original=True).
        - Robust Conv1D and LoRALinear support via base_layer weight transposition.
        - SVD and covariance eigendecomposition computed strictly in FP32 with NaN/Inf guards.
        - Unified critical-layer protection via should_compress_linear.
        - Adaptive rank retry when reconstruction error exceeds threshold.
        - Parameter reduction verification: ensures r * (in + out) < in * out.
        - Comprehensive per-layer statistics and logging.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target layers with activation covariance whitening and safe validation.

    `compress_linear(self, layer: nn.Module, cov: torch.Tensor | None = None, layer_name: str | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int, bool, bool]`
    :   Decompose a single linear, Conv1D, or LoRALinear layer via energy-thresholded SVD
        with covariance whitening, returning (svd_layer, relative_error, rank, used_cov).

`AdaptiveTransformerCompressor(model: nn.Module, energy_threshold: float = 0.92, min_rank: int = 16, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, use_jg_svd: bool = False, use_cholesky_jg: bool = True, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   Adaptive Ranks SVD Compressor.
        
        Dynamically determines ranks per layer according to cumulative singular value energy threshold:
            $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        with minimum rank floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor, \lfloor 0.10 \cdot d_{out} \rfloor)$,
        activation covariance whitening, and symmetrically distributed singular values:
            $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
        
        Production hardening features:
        - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
        - Safe fallback to original weights by default (fallback_to_original=True).
        - Robust Conv1D and LoRALinear support via base_layer weight transposition.
        - SVD and covariance eigendecomposition computed strictly in FP32 with NaN/Inf guards.
        - Unified critical-layer protection via should_compress_linear.
        - Adaptive rank retry when reconstruction error exceeds threshold.
        - Parameter reduction verification: ensures r * (in + out) < in * out.
        - Comprehensive per-layer statistics and logging.

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target layers with activation covariance whitening and safe validation.

    `compress_linear(self, layer: nn.Module, cov: torch.Tensor | None = None, layer_name: str | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int, bool, bool]`
    :   Decompose a single linear, Conv1D, or LoRALinear layer via energy-thresholded SVD
        with covariance whitening, returning (svd_layer, relative_error, rank, used_cov).

`AdaptiveSVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, use_fused_kernel: bool = False, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, lora_r: int | None = None, freeze_base: bool = False)`
:   Low-rank factorized linear layer:
        y = (x @ V.T) @ U.T + LoRA(x) * scaling + b
    Parameter count: r·(d_in + d_out) + LoRA params.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `use_lora: bool`
    :

    `weight: torch.Tensor`
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

    `freeze_base_weights(self) ‑> None`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`SVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, use_fused_kernel: bool = False, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, lora_r: int | None = None, freeze_base: bool = False)`
:   Low-rank factorized linear layer:
        y = (x @ V.T) @ U.T + LoRA(x) * scaling + b
    Parameter count: r·(d_in + d_out) + LoRA params.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `use_lora: bool`
    :

    `weight: torch.Tensor`
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

    `freeze_base_weights(self) ‑> None`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :