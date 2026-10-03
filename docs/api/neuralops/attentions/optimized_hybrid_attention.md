Module neuralops.attentions.optimized_hybrid_attention
======================================================
Optimized Hybrid Attention (OHA) — Production-Hardened (v2).

Fuses localized sliding-window self-attention with sequence-compressed
low-rank global attention via dynamic head-wise gated modulation and exact
pretrained weight transfer.

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_oha(layer=None, model=None, model_config=None, comp_config=None, layer_idx=0, total_layers=12, old_attn=None, **kwargs)`
:   

`inject_optimized_hybrid(layer=None, model=None, model_config=None, comp_config=None, layer_idx=0, total_layers=12, old_attn=None, **kwargs)`
:   

`inject_optimized_hybrid_attention(layer=None, model=None, model_config=None, comp_config=None, layer_idx=0, total_layers=12, old_attn=None, **kwargs)`
:   

Classes
-------

`LoRALinear(in_features, out_features, bias=True, lora_rank=16, lora_alpha=32.0, freeze_base=False, device=None, dtype=None)`
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

    `freeze_base_weights(self)`
    :

    `reset_parameters(self)`
    :

    `unfreeze_base_weights(self)`
    :

`OptimizedHybridAttention(config=None, rank=None, window_size=64, compression_rank=32, use_gate=True, gate_bias=2.0, pretrained_attn=None, old_attn=None, comp_config=None, is_decoder=None, **kwargs)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.