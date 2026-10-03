Module neuralops.attentions.gated_delta_net
===========================================
Gated DeltaNet Attention Module — Production-Hardened (v2).

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`chunked_delta_rule_scan(q, k, v, beta, alpha, chunk_size=32)`
:   Chunked Delta-Rule scan with O(L·chunk) intermediate memory.
    
    GD-FIX-09: no inner Python overhead beyond the minimal recurrence.

`inject_gated_delta_net(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   

Classes
-------

`GatedDeltaNet(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, old_attn: nn.Module | None = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k`
    :

    `W_q`
    :

    `W_v`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `query`
    :

    `v_lin`
    :

    `value`
    :

    ### Methods

    `freeze_base_weights(self)`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :

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

    `forward(self, x) ‑> Callable[..., typing.Any]`
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