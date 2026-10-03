Module neuralops.attentions.lora_attention
==========================================
LoRA Parameter-Efficient Fine-Tuning Attention Module (v2, Production-Hardened).

Replaces standard Q/K/V/O projections with frozen pretrained base weights plus
trainable low-rank residual adapters, with multi-architecture weight transfer,
GQA, RoPE, and causal decoding support.

This is intentionally an attention-swap module, NOT a compressor:
  - Base weights are frozen.
  - Only LoRA A/B are trainable.
  - Parameters are NOT reduced (compression_rate stays ~1.0).
  - Its purpose is PEFT, not compression.

Fixes (v2):
  LORA2-FIX-01  `actual_rank` is now purely the LoRA rank. `rank` (compression
                rank) is no longer conflated with LoRA rank.
  LORA2-FIX-02  Removed duplicated `if hasattr(layer, 'attention')` block.
  LORA2-FIX-03  Removed unreachable `elif hasattr(layer, 'attention')`.
  LORA2-FIX-04  Causal mask applied only for self-attention.
  LORA2-FIX-05  Robust `cfg = None` handling.
  LORA2-FIX-06  Proper `is_nested_encoder` propagation.
  LORA2-FIX-07  Idempotency guard in injector.
  LORA2-FIX-08  Storage accounting (`orig_storage_bytes`, `comp_storage_bytes`).
  LORA2-FIX-09  LoRA init: kaiming A, zeros B, dtype-safe.
  LORA2-FIX-10  `position_embeddings` accepted as explicit param + kwargs.
  LORA2-FIX-11  Strict mask handling with normalized valid-mask shape.
  LORA2-FIX-12  Deterministic injector recursion.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_lora_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`LoRAAttention(config: Any | None = None, model_config: Any | None = None, rank: int | None = None, lora_rank: int = 16, alpha: float = 32.0, dropout: float = 0.1, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   LoRA attention wrapper.
    
    NOTE: This module is deliberately a PEFT swap, NOT a compressor.
    `compression_rate` remains ~1.0; `num_compressed_layers` remains 0.
    Its role is to provide a trainable LoRA residual on top of frozen
    pretrained attention weights.
    
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

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer with frozen base weights and trainable LoRA adapters.
    
    Forward:
        y = x W^T + b + (alpha / r) * x A^T B^T
    where W, b are frozen, and A, B are trainable.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

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