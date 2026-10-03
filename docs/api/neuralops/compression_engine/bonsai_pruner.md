Module neuralops.compression_engine.bonsai_pruner
=================================================
Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradient Descent Recovery
& [CLS] Routing Protection. Registered in COMPRESSOR_REGISTRY as 'bonsai' and 'bonsai_pruner'.

Functions
---------

`bonsai_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory for Bonsai pruner.

`bonsai_prune_model(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory for Bonsai pruner.

Classes
-------

`BonsaiPrunedLoRALinear(in_features: int, out_features: int, bias: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear with sparse pruned base weight (frozen) + optional trainable LoRA adapter.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_linear(linear: nn.Linear | Any, pruned_weight: torch.Tensor, mask: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True) ‑> neuralops.compression_engine.bonsai_pruner.BonsaiPrunedLoRALinear`
    :

    ### Instance variables

    `effective_weight: torch.Tensor`
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

`BonsaiPruner(model: nn.Module, sparsity: float = 0.5, steps: int = 8, recovery_steps: int = 3, recovery_lr: float = 0.001, structured_24: bool = False, block_size: tuple[int, int] = (4, 4), protect_cls: bool = True, skip_attention_qkv: bool = True, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, adaptive_sparsity: bool = False, min_numel: int = 16, max_error: float | None = None, fallback_to_original: bool = True, recalibrate_ln: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, max_recovery_steps: int = 32, max_step_size: float = 0.01, **kwargs: Any)`
:   Bonsai Pruner: progressive multi-step iterative pruning with cubic schedule,
    inter-step gradient descent recovery, and [CLS]/GQA protection.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor