Module neuralops.attentions.svd_attention
=========================================
SVD-Compressed Sequence Attention.

This module reduces attention complexity from O(N²) to O(N·r)
by projecting K and V onto a rank-r subspace along the
sequence dimension.

CRITICAL LIMITATION — Read This:

    The projection basis is precomputed ONCE from the SVD of
    the key weight matrix W_K in copy_weights(). This means
    the projection does NOT depend on the input sequence —
    it is fixed at initialization time.

    A more principled approach would compute a per-input
    projection (e.g., via Linformer-style learned projections
    or Nyström landmark selection). The current implementation
    is a HEURISTIC and may not preserve attention semantics
    for all inputs.

    For CAUSAL decoders (GPT-2, Qwen, LLaMA), applying a
    sequence-axis SVD breaks causality because the projection
    combines future tokens into the subspace. The current
    implementation DISABLES SVD compression for causal mode
    and falls back to full attention. This means decoders
    get no speedup from this module.

Future Work:
    - Learnable input-dependent projection (Linformer-style)
    - Nyström landmark selection
    - Causal-aware projection with causal masking

Registered as: "svd_seq", "svd_sequence", "svd_sequence_attention".

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support.

`inject_svd_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_svd_linear_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
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

`SVDLinearAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 64, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float | None = 0.1, **kwargs: Any)`
:   Low-Rank SVD-Compressed Sequence Attention Mechanism.
    Hardened for production use across diverse transformer architectures.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `k_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :   Orthogonality diagnostic for the QR basis.
        Returns a scalar (no gradient). Use only for logging.

`LowRankSVDSequenceAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 64, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float | None = 0.1, **kwargs: Any)`
:   Low-Rank SVD-Compressed Sequence Attention Mechanism.
    Hardened for production use across diverse transformer architectures.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `k_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None, encoder_attention_mask: torch.Tensor | None = None, past_key_value: Any | None = None, output_attentions: bool | None = False, output_hidden_states: bool | None = False, return_dict: bool | None = True, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Must return (output, attention_weights_or_None).

    `get_ortho_loss(self) ‑> torch.Tensor`
    :   Orthogonality diagnostic for the QR basis.
        Returns a scalar (no gradient). Use only for logging.

`SVDAttention(config: Any | None = None, old_attn: torch.nn.modules.module.Module | None = None, rank: int | None = 64, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, dropout: float | None = 0.1, **kwargs: Any)`
:   Low-Rank SVD-Compressed Sequence Attention Mechanism.
    Hardened for production use across diverse transformer architectures.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `k_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `q_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_lin: neuralops.attentions.svd_attention.LoRALinear`
    :

    `v_proj: neuralops.attentions.svd_attention.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: torch.nn.modules.module.Module) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.

    `get_ortho_loss(self) ‑> torch.Tensor`
    :   Orthogonality diagnostic for the QR basis.
        Returns a scalar (no gradient). Use only for logging.