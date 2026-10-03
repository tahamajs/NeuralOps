Module neuralops.attentions.goldfinch_ahn
=========================================
GoldFinch-AHN: Adaptive Hybrid Network — Sliding Window + Gated Recurrent Memory.
Production-Hardened (v2).

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_goldfinch_ahn(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.goldfinch_ahn.GoldFinchAHNBlock | torch.nn.modules.module.Module | None`
:   

Classes
-------

`GatedRecurrentMemory(d_model: int)`
:   A light gated recurrent memory:
        g_t = sigmoid(W_g x_t + b_g)
        h_t = (1 - g_t) * h_{t-1} + g_t * tanh(W_x x_t + b_x)
        y_t = h_t
    
    Not a full DeltaNet but has the required gate structure and behaves as
    a proper recurrent memory (unlike the previous plain Linear layer).
    
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

`GoldFinchAHNBlock(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, window_size: int = 64, is_causal: bool = False, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, gate_init_bias: float = 1.8, comp_config: Any | None = None, **kwargs: Any)`
:   GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory.
    
    Semantics of `window_size`: full window width (in tokens). Effective
    radius = `window_size // 2`, so `window_size = 64` → ±32 tokens.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn, model_config=None, **kwargs)`
    :

    ### Instance variables

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `query`
    :

    `v_lin`
    :

    `v_proj`
    :

    `value`
    :

    ### Methods

    `freeze_base_weights(self)`
    :

    `storage_bytes(self) ‑> dict[str, int]`
    :

    `unfreeze_base_weights(self)`
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

    `unfreeze_base_weights(self)`
    :