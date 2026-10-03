Module neuralops.attentions.hybrid_attention
============================================
Hybrid softmax + linear attention with per-head gated fusion.

Valid attention REPLACEMENT (not a compressor).
Adds a small number of parameters (the gate) but reduces attention
computation at long sequence lengths by routing some heads to linear attention.

NOVELTY STATEMENT:
    Hybrid softmax + linear attention is NOT novel by itself
    (cf. MEGA 2023, BiGS 2024, Zipformer 2024). Our specific
    formulation differs in:

    1. PER-HEAD learned gate:
         gate_h = sigmoid(W_h · hidden_states)
       This allows different heads to prefer different attention types,
       in contrast to MEGA's single-head gate.

    2. Gate initialization to softmax-dominant (bias=10.0):
       Ensures the model starts IDENTICAL to the original softmax attention.
       The gate learns to route some heads to linear attention during training.

    3. No extra parameters beyond the gate:
       768 * 12 + 12 = 9,228 params (+0.014% on DistilBERT).

Bug-fix map:
  HA-FIX-01  out_proj always executed (no dead parameters in nested mode).
  HA-FIX-02  _linear_branch: proper causal + padding masking.
  HA-FIX-03  injector recursion guard.
  HA-FIX-04  injector double-injection guard.
  HA-FIX-05  removed 'comp_config' in locals() hack.
  HA-FIX-06  is_nested_encoder set correctly in injector.
  HA-FIX-07  KV-cache implemented (tuple + DynamicCache).
  HA-FIX-08  RoPE past_len offset + position_embeddings support.
  HA-FIX-09  head_mask broadcasting corrected (B, H, 1, 1).
  HA-FIX-10  _standard_branch: optional SDPA fast path.

Functions
---------

`_extract_valid_mask(mask: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device) ‑> torch.Tensor`
:   Robustly extract a boolean valid-token mask from any attention_mask format:
      - (B, N) int/bool     → padding mask
      - (B, N) float        → -inf/0 additive mask
      - (B, 1, N) or (B, 1, 1, N) broadcastable
      - (B, H, N, N) full attention matrix

`inject_smart_weighted_ensemble(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | Any | None`
:   

`inject_weighted_ensemble(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | Any | None`
:   

Classes
-------

`HybridAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Hybrid softmax + linear attention with per-head dynamic gated fusion.
    
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

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`WeightedEnsembleAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Hybrid softmax + linear attention with per-head dynamic gated fusion.
    
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

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, layer_idx: int = 0, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

`SmartWeightedEnsembleAttention(config: Any | None = None, rank: int | None = None, pretrained_attn: nn.Module | None = None, hidden_size: int | None = None, num_heads: int | None = None, head_dim: int | None = None, dropout: float = 0.1, is_decoder: bool | None = None, **kwargs: Any)`
:   Hybrid softmax + linear attention with per-head dynamic gated fusion.
    
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