Module neuralops.compression_engine.aa_svd_compressor
=====================================================
Activation-Aware SVD (AASVD) Compressor — Production-Hardened (v2).

Streaming EMA activation covariance accumulation:
    C_X^(t) = β·C_X^(t-1) + (1-β)·(X^T X) / B
Projects weights onto principal activation subspaces and replaces dense
nn.Linear/Conv1D layers with structural low-rank SVDLinear modules:
    y = (x · V^T) · U^T + LoRA(x) + b

Functions
---------

`aa_svd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   

`aa_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   

`aasvd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   

`aasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   

Classes
-------

`AASVDCompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 288, ema_beta: float = 0.9, max_error: float = 0.5, fallback_to_original: bool = True, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = True, force_rank: bool = False, lora_rank: int = 0, lora_alpha: float = 1.0, cov_dtype: str = 'float32', **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int | None = None, cov: torch.Tensor | None = None, cov_factors: tuple[torch.Tensor, torch.Tensor] | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
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