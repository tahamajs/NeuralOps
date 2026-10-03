Module neuralops.compression_engine.a3_compressor
=================================================
A3 Compressor: Activation-Aware Factorized SVD with Activation Energy
Weighting and LayerNorm Recalibration.

Registered as 'a3' and 'a3_compressor'.

Algorithm:
  1. Activation energy weighting (second moment E[x²]) — NOT Fisher.
     s_j = sqrt(E[x_j²] + eps), optionally normalized by RMS.
  2. Optional full Cholesky whitening of input covariance.
  3. STRS rank search: minimal rank satisfying the error bound.
  4. LayerNorm recalibration.
  5. Fallback to dense if error exceeds threshold.

Functions
---------

`a3_compressor_factory(model: nn.Module, comp_config: Any = None, rank: int = 64, use_fisher: bool = True, use_energy_weighting: bool | None = None, use_cholesky: bool = False, allow_ln_failure: bool = False, lora_rank: int = 0, lora_alpha: float = 1.0, recalibrate_ln: bool = True, **kwargs: Any)`
:   

Classes
-------

`A3Compressor(model: nn.Module, rank: int = 64, use_fisher: bool = True, use_energy_weighting: bool | None = None, use_cholesky: bool = False, allow_ln_failure: bool = False, lora_rank: int = 0, lora_alpha: float = 1.0, recalibrate_ln: bool = True, target_attention_only: bool = True, max_error: float = 0.5, max_retries: int = 5, fallback_to_original: bool = True, normalize_scale: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, n_calib: int = 500, device: str | torch.device | None = None, cov_dtype: str = 'float32', **kwargs: Any)`
:   A3: Activation-aware factorized SVD compressor.
    
    Uses E[x²] weighting (second moment), optional full Cholesky whitening,
    STRS rank search, and LayerNorm recalibration.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, module: nn.Module, name: str = '', cov: torch.Tensor | None = None, fisher: torch.Tensor | None = None, rank: int | None = None, activation_energy: torch.Tensor | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, X_for_error: torch.Tensor | None = None, **kwargs: Any)`
    :

`A3FactorizedLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = False)`
:   Low-rank factorized linear with optional LoRA:
        y = (x @ V.T) @ U.T + LoRA(x) + b
    Parameter count: r·(d_in + d_out) instead of d_in·d_out.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, rank: int, U_factor: torch.Tensor | None = None, V_factor: torch.Tensor | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = False) ‑> neuralops.compression_engine.a3_compressor.A3FactorizedLinear`
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