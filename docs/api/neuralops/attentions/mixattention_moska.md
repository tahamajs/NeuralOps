Module neuralops.attentions.mixattention_moska
==============================================
MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).
Production-Hardened (v2).

Functions
---------

`apply_rotary_pos_emb(q, k, cos, sin)`
:   

`fasa_inject(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_mixattention_moska(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
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

`MixAttentionMoSKA(config: Any | None = None, rank: int | None = 32, fasa_keep_ratio: float = 0.5, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention.
    
    For non-causal self-attention and encoder-style models, we prune the K/V
    set down to the top `fasa_keep_ratio` tokens by variance importance,
    always preserving the [CLS] token at index 0.
    
    For causal decoders, cross-attention, and incremental decoding (when
    `past_key_value` is present), we bypass pruning entirely and run exact
    dense attention via SDPA / manual softmax.
    
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

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> Callable[..., typing.Any]`
    :   Must return (output, attention_weights_or_None).

    `storage_bytes(self) ‑> dict[str, int]`
    :

`FASA(config: Any | None = None, rank: int | None = 32, fasa_keep_ratio: float = 0.5, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention.
    
    For non-causal self-attention and encoder-style models, we prune the K/V
    set down to the top `fasa_keep_ratio` tokens by variance importance,
    always preserving the [CLS] token at index 0.
    
    For causal decoders, cross-attention, and incremental decoding (when
    `past_key_value` is present), we bypass pruning entirely and run exact
    dense attention via SDPA / manual softmax.
    
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

    `storage_bytes(self) ‑> dict[str, int]`
    :