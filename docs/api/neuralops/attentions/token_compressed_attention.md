Module neuralops.attentions.token_compressed_attention
======================================================
Token-Compressed Attention — Production-Hardened (v2).

Compresses Key and Value sequence tokens using energy/norm + learned saliency
scoring, while always preserving token 0 ([CLS] / first token).

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin)`
:   

`inject_token_compressed_attn(layer=None, model=None, model_config=None, comp_config=None, layer_idx=0, total_layers=12, old_attn=None, **kwargs)`
:   

Classes
-------

`LoRALinear(in_features, out_features, bias=True, lora_rank=0, lora_alpha=1.0, freeze_base=True, device=None, dtype=None)`
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

    `forward(self, x) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`TokenCompressedAttention(config=None, rank=64, compression_ratio=0.5, pretrained_attn=None, lora_rank=None, lora_alpha=None, comp_config=None, is_decoder=None, **kwargs)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    `supports_causal`
    :   The type of the None singleton.

    `supports_gqa`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `o_proj`
    :

    `out_proj_alias`
    :

    `q_lin_alias`
    :