Module neuralops.compression_engine.replaceme_compressor
========================================================
ReplaceMe: Layer Removal and Replacement with Linear Residual Block.

Replaces the last N layers of a Transformer encoder or decoder with a compatible
LayerNorm + Linear residual block with optional trainable LoRA fine-tuning adapters.

Registered in COMPRESSOR_REGISTRY as 'replaceme', 'replaceme_compressor', and 'replaceme_factory'.

Functions
---------

`replaceme_compressor_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   ReplaceMe compressor factory function.

`replaceme_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   ReplaceMe compressor factory function.

Classes
-------

`ReplaceMeCompressor(model: nn.Module, remove_layers: int = 2, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, fallback_to_original: bool = True, device: str | torch.device | None = None, config: Any | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Replaces the trailing N layers of a Transformer with a lightweight
    LayerNorm + Linear residual block.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

`ReplacementBlock(hidden_size: int, lora_r: int = 0, lora_alpha: float = 1.0, return_tuple: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Drop-in replacement for a transformer layer.
    
    Preserves tensor shapes and maintains signal propagation via a residual
    connection with optional LoRA fine-tuning adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | Any, *args: Any, **kwargs: Any) ‑> torch.Tensor | tuple[torch.Tensor, ...]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.