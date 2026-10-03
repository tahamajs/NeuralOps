Module neuralops.attentions.nasc_attention
==========================================
NASC: NEXUS-AEGIS Spectral Cascade Attention.
Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.

NOTE: This module is LoRA-only. The rank argument is accepted but not used for
SVD compression. Use lora_rank to control adapter size.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_nasc(layer: torch.nn.modules.module.Module | None = None, model: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 1, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.nasc_attention.NASCAttention | torch.nn.modules.module.Module | None`
:   

`inject_nasc_attention(layer: torch.nn.modules.module.Module | None = None, model: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 1, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.nasc_attention.NASCAttention | torch.nn.modules.module.Module | None`
:   

`softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   Stable softpick function with negative-clamp and softmax fallback.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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

    `reset_parameters(self) ‑> None`
    :

`NASCAttention(config: Any, rank: int | None = 16, window_size: int = 64, block_size: int = 32, top_k: int = 2, rff_features: int = 64, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   NASC: NEXUS-AEGIS Spectral Cascade Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: neuralops.attentions.nasc_attention.LoRALinear`
    :

    `query: neuralops.attentions.nasc_attention.LoRALinear`
    :

    `value: neuralops.attentions.nasc_attention.LoRALinear`
    :