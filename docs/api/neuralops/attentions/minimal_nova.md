Module neuralops.attentions.minimal_nova
========================================

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_minimal_nova(layer: torch.nn.modules.module.Module | None = None, model: torch.nn.modules.module.Module | None = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 1, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.minimal_nova.MinimalNOVAAttention | torch.nn.modules.module.Module | None`
:   Injection hook for MinimalNOVAAttention into transformer layer or entire model.

Classes
-------

`FactorisedLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, energy: float = 0.95, lora_rank: int = 16, lora_alpha: float = 32.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
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
    :   SVD-Factorized Linear Transformation for NOVA Architecture.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None, energy: float | None = None) ‑> None`
    :

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

`MinimalNOVAAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 0.95, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MinimalNOVA: SVD-Factorized Low-Rank Projection Attention.
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

    `key: neuralops.attentions.minimal_nova.FactorisedLinear`
    :

    `query: neuralops.attentions.minimal_nova.FactorisedLinear`
    :

    `value: neuralops.attentions.minimal_nova.FactorisedLinear`
    :

    ### Methods

    `get_ortho_loss(self) ‑> torch.Tensor`
    :