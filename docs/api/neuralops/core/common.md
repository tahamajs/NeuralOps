Module neuralops.core.common
============================
Unified shared primitives and utility layers for attention and compression modules.
Centralizes repeatedly copied helpers (_cfg_get, repeat_kv, _detect_decoder, LoRALinear, FactorizedLinear).

Functions
---------

`_apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) to Query and Key tensors.

`_cfg_get(config: Any, key: str, default: Any = None) ‑> Any`
:   Safely retrieves a configuration attribute from a dict, object, or None.
    Handles nested fallbacks without throwing AttributeErrors.

`_detect_decoder(config: Any = None, is_decoder: bool | None = None, old_attn: nn.Module | None = None) ‑> bool`
:   Detects whether an attention layer should operate in causal decoder mode.

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Repeats Key/Value heads along the head dimension for Grouped-Query Attention (GQA).
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim)
    Returns: (batch, num_attention_heads, seqlen, head_dim)

Classes
-------

`FactorizedLinear(in_features: int, out_features: int, rank: int = 128, lora_rank: int = 32, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Decomposed linear layer: W ≈ W2 @ W1 with optional LoRA residual branch.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
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

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :

`LoRALinear(base_or_in_features: nn.Module | int, out_features: int | None = None, bias: bool = True, lora_rank: int = 16, r: int | None = None, lora_alpha: float = 32.0, lora_dropout: float = 0.0, freeze_base: bool = False, merge_weights: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Universal LoRA linear layer:
    Supports both wrapper mode: LoRALinear(base_layer, r=16, ...)
    and standalone mode: LoRALinear(in_features, out_features, bias=True, lora_rank=16, ...)
    
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

    `merge(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

    `unmerge(self) ‑> None`
    :