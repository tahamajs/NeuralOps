Module neuralops.attentions.gated_deltanet_block
================================================
Gated DeltaNet Block with Optional Multi-Head Attention Bypass.

Implements an efficient hybrid linear-recurrent block switching dynamically between standard
multi-head self-attention and Bidirectional Gated DeltaNet with pretrained weight transfer.

Functions
---------

`inject_gated_deltanet_block(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`GatedDeltaNetBlock(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, num_heads: int | None = None, use_full_attention: bool = False, comp_config: Any | None = None, **kwargs: Any)`
:   Gated DeltaNet Transformer Block with Pretrained Weight Transfer.
    
    Mathematical Formulation:
        $$\text{Output} = \begin{cases}
            \text{LayerNorm}(X + \text{MHA}(X, X, X)), & \text{if } \text{use\_full\_attention} \\
            \text{LayerNorm}(X + \text{GatedDeltaNet}(X)), & \text{otherwise}
        \end{cases}$$
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: Any`
    :

    `k_proj: Any`
    :

    `key: Any`
    :

    `out_lin: Any`
    :

    `out_proj: Any`
    :

    `q_lin: Any`
    :

    `q_proj: Any`
    :

    `query: Any`
    :

    `v_lin: Any`
    :

    `v_proj: Any`
    :

    `value: Any`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained weights to either MHA or GatedDeltaNet.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Forward pass for GatedDeltaNetBlock.