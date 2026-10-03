Module neuralops.compression_engine.nmf_compressor
==================================================
NMF: Non-negative Matrix Factorization Linear Compressor.

Registered in COMPRESSOR_REGISTRY as 'nmf', 'nmf_compressor'.

Functions
---------

`nmf_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   

Classes
-------

`NMFCompressor(model: nn.Module | None = None, rank: int = 64, max_iter: int = 100, lora_r: int = 0, lora_alpha: float = 1.0, cfg: Any | None = None, target_layers: list[str] | None = None, target_attention_only: bool = False, max_error: float = 0.7, fallback_to_original: bool = True, **kwargs: Any)`
:   NMF Compressor: decomposes linear layer magnitudes into non-negative
    low-rank factors, preserving the sign pattern of the original weight.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, linear: nn.Module) ‑> neuralops.compression_engine.nmf_compressor.NMFLinear`
    :

`NMFLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Linear layer with weight factorized via sign-aware NMF:
    
        W = H @ V * W_sign
    
    where H ≥ 0, V ≥ 0, and W_sign ∈ {-1, +1} stores the sign of the original
    weight. Since `torch.sign` cannot be reconstructed by NMF, we factor the
    magnitude |W| with non-negative factors and remember the sign separately.
    
    Storage: H (out × r), V (r × in), W_sign (out × in as int8 or fp16).
    Note: W_sign storage is non-trivial (out × in elements). If storage is
    the priority, use a signed-NMF variant that avoids storing W_sign.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Module, rank: int = 64, max_iter: int = 30, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.nmf_compressor.NMFLinear`
    :   Fit sign-aware NMF on the given Linear / Conv1D / LoRALinear module
        using multiplicative updates (Lee & Seung 2001) with NNDSVD init.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix: H @ V * W_sign.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_weight(self) ‑> torch.Tensor`
    :