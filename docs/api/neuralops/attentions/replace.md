Module neuralops.attentions.replace
===================================
Model Attention Replacement and Dynamic Graph Surgery Utilities — v2.

Functions
---------

`_cfg_get(cfg: Any, key: str, default: Any = None) ‑> Any`
:   

`_get_valid_mask(attention_mask: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device) ‑> torch.Tensor`
:   

`_replace_bert_attention(layer: torch.nn.modules.module.Module, inject_fn: Any, model_config: Any = None, **kwargs: Any) ‑> None`
:   

`_replace_distilbert_attention(layer: torch.nn.modules.module.Module, inject_fn: Any, model_config: Any = None, **kwargs: Any) ‑> None`
:   

`_replace_gpt_attention(layer: torch.nn.modules.module.Module, inject_fn: Any, model_config: Any = None, **kwargs: Any) ‑> None`
:   

`build_attention(attention_name: str, hidden_size: int, num_heads: int, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   

`inject_replace(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe injection of ReplaceAttention with recursion guard.

`inject_replace_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe injection of ReplaceAttention with recursion guard.

`replace_attention_in_model(model: torch.nn.modules.module.Module, attention_name: str, attention_params: dict[str, typing.Any] | None = None) ‑> torch.nn.modules.module.Module`
:   

Classes
-------

`ReplaceAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, num_attention_heads: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Robust identity / low-rank SVD replacement attention.
    
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

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, output_attentions: bool = False, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`Replace(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, num_attention_heads: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Robust identity / low-rank SVD replacement attention.
    
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