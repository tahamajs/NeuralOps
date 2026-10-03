Module neuralops.core.replace_attention
=======================================
Model Attention Replacement and Dynamic Graph Surgery Utilities.

Bug-fix map (2026 revision):
  RE-01   Single registry key 'replace'/'replace_class' separation.
  RE-02   _replace_distilbert_attention wraps injector in try/except.
  RE-03   Dynamic config via SimpleNamespace (not type()).
  RE-04   build_attention via inspect.signature (no blind TypeError catching).
  RE-05   inspect.signature for arg detection (not getfullargspec).
  RE-06   layer.attention.self access guarded.
  RE-07   ReplaceAttention now supports optional low-rank factorization.
  RE-08   is_nested_encoder consistently set in injector.
  RE-09   _get_valid_mask robust for 4D masks.
  RE-10   SDPA receives attention_mask.
  RE-11   Every except logs at debug level.
  RE-12   out_lin / o_proj / dense aliases added.
  RE-13   num_heads=0 guarded.
  RE-14   Weight copy shape-checked.
  RE-15   Qwen / LLaMA architectures supported.
  RE-16   Shared _replace_layers helper (no duplicate logic).
  RE-17   Registry lookup validates callability.
  RE-18   _cfg_get defined at module scope (fixes Pylance error).
  RE-19   _FactorizedLinear exposes .bias property.
  RE-20   _replace_layers filters attention_params to safe keys.
  RE-21   SDPA causal disabled when additive mask is present.
  RE-22   copy_weights supports GQA when hidden_size != all_head_size.

Functions
---------

`_cfg_get(cfg: Any, key: str, default: Any = None) ‑> Any`
:   Safely retrieve attribute from dict, object, namespace, or None.

`_get_valid_mask(attention_mask: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device) ‑> torch.Tensor`
:   Return (B, seq_len) boolean valid mask.

`build_attention(attention_name: str, hidden_size: int, num_heads: int, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Public alias.

`inject_replace(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe injection of ReplaceAttention with recursion guard.

`inject_replace_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | None`
:   Safe injection of ReplaceAttention with recursion guard.

`replace_attention_in_model(model: torch.nn.modules.module.Module, attention_name: str, attention_params: dict[str, typing.Any] | None = None) ‑> torch.nn.modules.module.Module`
:   Replace all self-attention layers in a transformer model with target attention variant.

Classes
-------

`ReplaceAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, num_attention_heads: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Robust identity / low-rank SVD replacement attention.
    Verified multi-architecture weight copying and valid forward pass.
    
    If `rank > 0`, projections are SVD-factorized for parameter reduction.
    If `rank == 0` or None, uses full nn.Linear (identity-baseline behavior).
    
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
    :   Universal weight extraction for all supported architectures.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, output_attentions: bool = False, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`Replace(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, hidden_size: int | None = None, num_heads: int | None = None, num_attention_heads: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Robust identity / low-rank SVD replacement attention.
    Verified multi-architecture weight copying and valid forward pass.
    
    If `rank > 0`, projections are SVD-factorized for parameter reduction.
    If `rank == 0` or None, uses full nn.Linear (identity-baseline behavior).
    
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
    :   Universal weight extraction for all supported architectures.