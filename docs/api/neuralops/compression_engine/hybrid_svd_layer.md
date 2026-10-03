Module neuralops.compression_engine.hybrid_svd_layer
====================================================
Hybrid SVD Layer: weight-based SVD compression for attention/linear layers.

Registered under:
  - 'hybrid_svd'
  - 'hybrid_svd_layer'
  - 'direct_svd_encoder'

The deprecated SVDSingleHead (which performed SVD on live activations) is
retained only for backward compatibility and emits a warning on use.

Functions
---------

`direct_svd_encoder_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Hybrid SVD factory function.

`hybrid_svd_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Hybrid SVD factory function.

`hybrid_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Hybrid SVD factory function.

Classes
-------

`DirectSVDEncoderLayer(config: Any | None = None, hidden_size: int | None = None, intermediate_size: int | None = None, num_attention_heads: int | None = None, svd_rank: int = 20, layer_norm_eps: float = 1e-12, hidden_dropout_prob: float = 0.1, lora_r: int = 0, lora_alpha: float = 1.0)`
:   Encoder layer that replaces MHA with SVDSingleHead + standard FFN.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, output_attentions: bool = False, *args: Any, **kwargs: Any) ‑> torch.Tensor | tuple[torch.Tensor, ...]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`HybridSVDCompressor(model: nn.Module, svd_rank: int = 128, use_full_layer_replacement: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.85, target_attention_only: bool = False, fallback_to_original: bool = True, cfg: Any | None = None, target_modules: list[str] | None = None, **kwargs: Any)`
:   Weight-based SVD compressor for Linear / Conv1D layers.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Decompose linear layers via SVD and replace with SVDLinear.

`SVDSingleHead(hidden_size: int, svd_rank: int = 20)`
:   DEPRECATED single-head SVD attention.
    
    Performing SVD on live activations is O(N^2 d), numerically fragile, and
    theoretically incorrect as an attention replacement. Use the weight-based
    HybridSVDCompressor instead.
    
    Kept strictly for backwards compatibility. First use emits a warning.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, hidden_states: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.