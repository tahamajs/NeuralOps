Module neuralops.compression_engine.tomoe_compressor
====================================================
ToMoE: converts dense linear and Conv1D layers (e.g. FFN and attention projections)
into a Mixture of Experts using weight clustering.
Registered as 'tomoe' and 'tomoe_compressor'.

Functions
---------

`inject_tomoe(model: Any = None, **kwargs: Any) ‑> Any`
:   Safe injection hook for ToMoE.

`tomoe_compressor_factory(model: nn.Module, comp_config: Any = None, num_experts: int = 4, top_k: int = 2, **kwargs: Any) ‑> Any`
:   Factory function for ToMoE compressor.

`tomoe_factory(model: nn.Module, comp_config: Any = None, num_experts: int = 4, top_k: int = 2, **kwargs: Any) ‑> Any`
:   Factory function for ToMoE compressor.

Classes
-------

`ToMoECompressor(model: nn.Module, num_experts: int = 4, top_k: int = 2, max_error: float = 0.2, lora_r: int = 0, lora_alpha: float = 16.0, freeze_experts: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, **kwargs: Any)`
:   ToMoE Compressor: dynamically replaces dense feed-forward and projection layers
    with an MoE-fied equivalent.
    
    Args:
        model: PyTorch model.
        num_experts: Number of expert sub-layers to form (default: 4).
        top_k: Number of experts to route each token to (default: 2).
        max_error: Maximum allowed relative Frobenius error on reconstruction (default: 0.20).
        lora_r: LoRA adapter rank on experts and router (0 = disabled).
        lora_alpha: LoRA scaling factor (default: 16.0).
        freeze_experts: Whether to freeze expert base weights (default: True).
        target_attention_only: Whether to restrict to attention layers only.
        target_modules: Optional list of target layer name substrings.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `PROTECTED_KEYWORDS`
    :   The type of the None singleton.

    ### Methods

    `compress(self, orig_model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress target layers into ToMoE Mixture of Experts layers with full error validation.

`ToMoELinear(original: nn.Module | None = None, in_features: int | None = None, out_features: int | None = None, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, num_experts: int = 4, top_k: int = 2, lora_r: int = 0, lora_alpha: float = 16.0, freeze_experts: bool = True, max_error: float = 0.2)`
:   A single MoE-fied linear layer: learned routing + expert sub-layers.
    
    Features:
    - Clusters output neurons using FP32 KMeans (with equal-division fallback).
    - Vectorized forward pass using gather and scatter_add_ for equal-sized experts,
      or vectorized index-dispatch for variable-sized experts.
    - Numerical stability in FP16, BF16, and FP32.
    - Exposes reconstructed effective weight and bias properties for validation.
    - Supports optional trainable LoRA adapters on experts and router.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `bias: torch.Tensor | None`
    :   Reconstructed effective full bias vector (out_features,).

    `masks: list[torch.Tensor]`
    :

    `weight: torch.Tensor`
    :   Reconstructed effective full weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.