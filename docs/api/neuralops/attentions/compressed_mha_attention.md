Module neuralops.attentions.compressed_mha_attention
====================================================
Compressed Multi-Head Attention for DistilBERT and Multi-Architecture via Joint Truncated SVD.

Performs joint singular value decomposition on stacked query, key, and value projection matrices
as well as the final output projection matrix with frozen base parameters, trainable LoRA residual adapters,
GQA, RoPE, causal decoding, and full pretrained weight transfer.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`compressed_mha_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, **kwargs: Any) ‑> torch.Tensor`
:   

`inject_compressed_mha(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`CompressedDistilBertAttention(model_config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, dropout: float = 0.1, lora_rank: int = 16, pretrained_attn: nn.Module | None = None, config: Any | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
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

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Instance variables

    `out_weight: torch.Tensor`
    :   Reconstruct the effective base Out weight from SVD factors.

    `qkv_weight: torch.Tensor`
    :   Reconstruct the effective base QKV weight from SVD factors.

    ### Methods

    `reset_lora_parameters(self) ‑> None`
    :