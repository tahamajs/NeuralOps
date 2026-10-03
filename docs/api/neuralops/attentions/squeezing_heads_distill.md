Module neuralops.attentions.squeezing_heads_distill
===================================================
Squeezing Heads Attention Distillation Layer (v2, Real Head Compression).

Multi-head attention wrapper that truly compresses Q/O projections by
keeping only the top-k most important heads, selected from pretrained Q/K/V.

Fixes (v2):
  SQ2-FIX-01  Real head compression: Q and O are sized to num_kept_heads,
              not num_heads. `head_keep_ratio` is derived from `rank`
              when not explicitly provided.
  SQ2-FIX-02  Head importance scoring from pretrained weights:
              score(h) = ||W_q[h]||_F + ||W_k[h]||_F + ||W_v[h]||_F.
              Top-k heads are selected per layer.
  SQ2-FIX-03  copy_weights copies only kept head slices.
  SQ2-FIX-04  KV projections preserve all num_kv_heads (GQA-compatible).
              Kept Q heads are matched to KV heads via GQA grouping.
  SQ2-FIX-05  Honest compression reporting: `compression_rate` reflects
              real parameter reduction, not LoRA addition.
  SQ2-FIX-06  Nested encoder detection unified.
  SQ2-FIX-07  Causal mask offset-aware.
  SQ2-FIX-08  SDPA fast path with correct attn_mask + is_causal.
  SQ2-FIX-09  Strict output dim check.
  SQ2-FIX-10  No silent `min()` truncation — shape mismatches raise.
  SQ2-FIX-11  Idempotency guard.
  SQ2-FIX-12  head_mask broadcasting safe.
  SQ2-FIX-13  Dropout applied before head_mask (matches HF).

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_squeezing_heads(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`SqueezingHeadsDistiller(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, head_keep_ratio: float | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.
    
    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim
    
    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.
    
    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    
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

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`SqueezingHeads(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, head_keep_ratio: float | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.
    
    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim
    
    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.
    
    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    
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

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`SqueezingHeadsAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = None, head_keep_ratio: float | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.
    
    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim
    
    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.
    
    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    
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