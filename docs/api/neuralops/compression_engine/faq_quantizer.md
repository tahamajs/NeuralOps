Module neuralops.compression_engine.faq_quantizer
=================================================
FAQ (Feature-Activation-Aware Quantization): adjusts quantization scales using
activation statistics. Registered as 'faq'.

Functions
---------

`faq_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   FAQ compressor factory function.

Classes
-------

`FAQLinear(in_features: int, out_features: int, bias: bool = True, bits: int = 8, symmetric: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   Linear with FAQ activation-aware quantization, STE backprop and optional LoRA.

    ### Static methods

    `from_linear(module: nn.Linear | Any, act_scale: torch.Tensor | None = None, bits: int = 8, symmetric: bool = True, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.faq_quantizer.FAQLinear`
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

    `get_weight(self) ‑> torch.Tensor`
    :

`FAQQuantizer(model: nn.Module, tokenizer: Any | None = None, bits: int = 8, calibration_texts: list[str] | None = None, calibration_data: Any | None = None, alpha: float = 0.5, symmetric: bool = True, use_ste: bool = True, lora_r: int = 16, lora_alpha: float = 1.0, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, skip_attention_qkv: bool = False, cfg: Any | None = None, num_calibration_batches: int = 8, **kwargs: Any)`
:   FAQQuantizer: Feature-Activation-Aware Quantization.
    
    Collects activation statistics to scale channel-wise quantization step sizes,
    minimizing activation-weighted reconstruction distortion.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor