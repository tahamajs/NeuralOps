Module neuralops.attentions.compressed_attention
================================================
Compressed Multi-Head Attention via Truncated Singular Value Decomposition (SVD).

Provides rank-constrained and singular-energy-thresholded low-rank approximations of standard
PyTorch attention input (QKV) and output projection matrices with full pretrained weight transfer,
trainable LoRA residual adapters, GQA, RoPE, and causal decoding support.

Functions
---------

`inject_compressed_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`CompressedAttention(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer2

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

    `copy_weights(self, src: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `reset_lora_parameters(self) ‑> None`
    :

`CompressedAttentionLayer(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer2

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

`CompressedAttentionLayer2(config: Any | None = None, rank: int | None = None, rank_qkv: int = 256, rank_out: int = 256, lora_rank: int = 16, original_layer: Any = None, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.compressed_attention.CompressedAttentionLayer
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module