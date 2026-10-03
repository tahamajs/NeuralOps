Module neuralops.attentions.mamba_block
=======================================
Faithful Mamba Block with SVD-based Attention Weight Distillation.

Based on Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (arXiv:2312.00752).

Functions
---------

`inject_mamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | Any | None`
:   

Classes
-------

`MambaBlock(config: Any | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, is_decoder_override: bool | None = None, freeze_base: bool = False, **kwargs: Any)`
:   Faithful S6 / Mamba block, production-hardened.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: nn.Module) ‑> None`
    :   Distill attention weights (W_q, W_k, W_v, W_o) into Mamba's S6
        parameters via low-rank mapping:
            W_v  → in_proj (SSM input path)
            W_k  → x_proj (B branch, via SVD)
            W_q  → x_proj (C branch, via SVD)
            W_o  → out_proj
            dt   → kept at initialization (learnable, input-dependent via x_proj)

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, *args: Any, past_state: Any | None = None, past_key_value: Any | None = None, output_attentions: bool = False, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :   MB-04: match the framework's LoRA-freeze convention.

    `unfreeze_base_weights(self) ‑> None`
    :

`Mamba(config: Any | None = None, d_model: int | None = None, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None, conv_bias: bool = True, bias: bool = False, bidirectional: bool = False, is_decoder_override: bool | None = None, freeze_base: bool = False, **kwargs: Any)`
:   Faithful S6 / Mamba block, production-hardened.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: nn.Module) ‑> None`
    :   Distill attention weights (W_q, W_k, W_v, W_o) into Mamba's S6
        parameters via low-rank mapping:
            W_v  → in_proj (SSM input path)
            W_k  → x_proj (B branch, via SVD)
            W_q  → x_proj (C branch, via SVD)
            W_o  → out_proj
            dt   → kept at initialization (learnable, input-dependent via x_proj)

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, *args: Any, past_state: Any | None = None, past_key_value: Any | None = None, output_attentions: bool = False, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :   MB-04: match the framework's LoRA-freeze convention.

    `unfreeze_base_weights(self) ‑> None`
    :