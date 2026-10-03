Module neuralops.attentions.hgrn2_layer
=======================================
Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer.

Implements bidirectional/causal element-wise linear recurrence with dynamic input-dependent
log-space forget gating, input modulation, LoRA adapters, and multi-architecture pretrained weight transfer.

Functions
---------

`inject_hgrn2_layer(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any) ‑> Any | None`
:   Safely constructs and injects HGRN2Layer into a model or transformer layer.

Classes
-------

`HGRN2Layer(config: Any | None = None, rank: int | None = None, d_model: int | Any | None = None, res_clip: float = 30.0, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, freeze_base: bool = True, **kwargs: Any)`
:   Hierarchically Gated Recurrent Network v2 (HGRN-2) Layer.
    Supports causal mode for decoders, bidirectional mode for encoders, LoRA adapters,
    and pretrained weight copying from DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    
    NOTE: This is NOT a compression module. It replaces attention with an RNN.
    Parameter count is the same. Only use for speed improvements on
    long sequences (when parallel scan is implemented).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `key: nn.Module`
    :

    `query: nn.Module`
    :

    `value: nn.Module`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained projection weights across all 4 architectures.