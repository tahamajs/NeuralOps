Module neuralops.attentions.svd_linear_attention
================================================
SVD Factorized Linear Attention & Kernelized Linear Attention
with Exponential Decay Gating — Production-Hardened (v2).

Hardened for production use across:
  - DistilBERT (Flat MHA)
  - RoBERTa / BERT (Nested Encoder Self-Attention)
  - GPT-2 (Causal Decoder with Conv1D)
  - Qwen / LLaMA (GQA with Rotary Position Embeddings)

Fixes (v2):
  SVDLA-FIX-01  `copy_weights` no longer raises on missing projections.
                It logs a warning and falls back to random init so the
                injection never crashes the whole pipeline.
  SVDLA-FIX-02  Near-zero Frobenius norm triggers a warning, not a raise.
  SVDLA-FIX-03  `triton_dispatch` is wrapped in try/except with fallback
                to plain F.linear. This eliminates the STATUS_ACCESS_VIOLATION
                (0xC0000005) crash at high ranks.
  SVDLA-FIX-04  Idempotency guard in injector: refuses to re-wrap.
  SVDLA-FIX-05  Honest storage accounting.
  SVDLA-FIX-06  Output padding/truncation hack replaced with an explicit
                RuntimeError when shapes mismatch (should never happen).
  SVDLA-FIX-07  `chunk_size` configurable, default 64.
  SVDLA-FIX-08  Decay gate bias clipped to a finite range.
  SVDLA-FIX-09  `mode` is normalized once; kernel selection is unambiguous.
  SVDLA-FIX-10  Single mask-normalization helper.
  SVDLA-FIX-11  `log_lam_c` clamp floor raised to avoid NaN.
  SVDLA-FIX-12  `is_custom_attention = False` on `SVDLinear` (it is not
                an attention module, just a factorized linear).
  SVDLA-FIX-13  Registry declares `supports_bidirectional`.
  SVDLA-FIX-14  `step_recurrent` respects `use_decay_gating` cleanly.
  SVDLA-FIX-15  `set_low_rank_factors` respects out/in dimension clamps.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`compute_exponential_decay(decay_logits: torch.Tensor, activation: str = 'softplus') ‑> torch.Tensor`
:   

`inject_svd_linear_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, rank: int | None = 192, mode: str = 'exact', **kwargs: Any) ‑> Any | None`
:   

Classes
-------

`SVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, use_fused_kernel: bool = False, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Low-rank factorized linear layer:
        y = (x V^T) U^T + b + (alpha / r_lora) (x A^T) B^T
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
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

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`SVDLinearAttention(config: Any | None = None, old_attn: nn.Module | None = None, rank: int | None = None, mode: str = 'exact', freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, use_decay_gating: bool = False, decay_activation: str = 'softplus', decay_channel_wise: bool = False, use_fused_kernel: bool = False, use_rms_norm: bool = False, chunk_size: int = 64, **kwargs: Any)`
:   SVD Factorized Linear Attention with optional exponential decay gating.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.projection_factorized_attn.ProjectionFactorizedAttention
    * neuralops.attentions.svd_linear_decay.DecayGatedLinearAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: nn.Module) ‑> None`
    :   SVDLA-FIX-01: never raise. On missing projections, log a warning
        and fall back to the randomly-initialized state.

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :

    `step_recurrent(self, x_t: torch.Tensor, past_state: tuple[torch.Tensor, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]`
    :