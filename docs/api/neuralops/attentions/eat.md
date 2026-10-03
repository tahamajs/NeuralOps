Module neuralops.attentions.eat
===============================
EAT: Efficient Attention Transformer with Progressive Token Pruning.
Production-Hardened (v2).

Fixes (v2):
  EAT-FIX-01  Removed output padding/truncation hack.
  EAT-FIX-02  RoPE cos/sin broadcast corrected.
  EAT-FIX-03  `is_nested_encoder` no longer triggers for DistilBERT.
  EAT-FIX-04  Idempotency guard in injector.
  EAT-FIX-05  SDPA causal + mask combination corrected (broadcast-safe).
  EAT-FIX-06  `head_mask` applied uniformly across dense and SDPA paths.
  EAT-FIX-07  `ProgressiveTokenPruning` only built when explicitly requested
              via `enable_pruning=True` or `keep_ratio` in `comp_config`
              (no more silent construction from `config.keep_ratio`).
  EAT-FIX-08  `LoRALinear.forward` dtype-safe.
  EAT-FIX-09  Storage accounting.
  EAT-FIX-10  Config None-safety in `__init__`.
  EAT-FIX-11  `copy_weights` device/dtype-safe and shape-tolerant.
  EAT-FIX-12  Fallback GQA pooling uses `repeat_interleave`.
  EAT-FIX-13  `logger` used consistently (no `print`).
  EAT-FIX-14  `_make_placeholder` for missing source modules.
  EAT-FIX-15  Warnings (not runtime errors) for near-zero weights.

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`
:   

`inject_eat(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`EATBlock(config: Any | None = None, rank: int | None = 64, d_model: int | Any = 768, n_heads: int = 12, prune_layer: Any | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    `k_proj`
    :

    `key`
    :

    `o_proj`
    :

    `out_proj`
    :

    `q_proj`
    :

    `query`
    :

    `v_proj`
    :

    `value`
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