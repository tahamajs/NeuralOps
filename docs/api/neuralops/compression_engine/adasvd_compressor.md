Module neuralops.compression_engine.adasvd_compressor
=====================================================
AdaSVD: Adaptive SVD with Activation Whitening, Two-Sided Weighted ALS,
and Mean-Shift Bias Absorption.

Registered as 'adasvd', 'adasvd_compressor', 'adasvd_factory',
'adasvd_compressor_factory'.

Key components:
  1. Mean-Shift Bias Absorption:
       b_new = b + W @ mu_x
       y = A B (x - mu_x) + b_new
  2. Two-Sided Weighted ALS:
       minimize || D^{1/2} (W - AB) C^{1/2} ||_F^2
       with C = C_X and D = diag(Var(Y)).
  3. Information-Theoretic Effective Rank:
       erank = exp(-sum(p_i * log p_i)), p_i = s_i^2 / sum(s_j^2)
  4. Fused Q/K/V splitting for GPT-2 c_attn (Conv1D).

Functions
---------

`adasvd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

`adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

`AdaSVDCompressor(model: nn.Module, rank: int | None = None, target_param_ratio: float = 0.7, num_iters: int = 10, max_error: float = 0.5, fallback_to_original: bool = True, target_attention_only: bool = True, min_rank: int = 16, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_texts: list[str] | None = None, use_weighted_als: bool = False, use_mean_shift: bool = True, use_gating_d: bool = True, device: str | torch.device | None = None, comp_config: Any = None, max_calib_tokens: int = 4096, fused_qkv_error_threshold: float = 0.3, gelu_layer_patterns: tuple[str, ...] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int, name: str = '', covariance: torch.Tensor | None = None, X_calib: torch.Tensor | None = None, out_variance: torch.Tensor | None = None, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module | None, float, int]`
    :

`FusedQKVSVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Compressed replacement for GPT-2's fused c_attn.
    
    Stores three independent low-rank factors (Q, K, V). Forward concatenates
    them so the output shape matches the original fused projection.
    
    Supports mean-shift bias absorption: if `_input_mean` is set, the input
    is centered before the low-rank projection and the shift is added back
    through the bias.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstruct the fused (in, out) Conv1D weight for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `set_fused_factors(self, W_fused: torch.Tensor) ‑> None`
    :   Split a fused (in, out) or (out, in) weight into Q/K/V factors.

    `set_input_mean(self, mu: torch.Tensor) ‑> None`
    :   Register the calibration activation mean (input centering).