Module neuralops.attentions.unified_rope
========================================
Universal Rotary Position Embedding Attention (UnifiedRoPE).
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support.

`inject_unified_rope(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 0, lora_alpha: float = 1.0, freeze_base: bool = True)`
:   Linear projection with frozen base weights and trainable LoRA adapters.
    
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

`UnifiedRoPE(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, max_seq_len: int = 4096, base: int = 10000, dropout: float = 0.1, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Universal Rotary Position Embedding Attention module.
    
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

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor`
    :   Forward pass computing RoPE attention, returning (output, attention_weights).