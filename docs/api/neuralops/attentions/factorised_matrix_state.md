Module neuralops.attentions.factorised_matrix_state
===================================================
Factorised Matrix State Attention Module.

Implements multi-head bidirectional/causal associative matrix memory updates combined with
learnable decay gating, denominator normalization, low-rank factorized output projection,
and pretrained weight transfer.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models (forward-only recurrence)
- Nested encoder detection to bypass output projection in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- True O(L) linear complexity recurrence without O(L^2) memory footprint
- Numerically stable FP32 state accumulation with nan_to_num protection
- Safe injection factory returning None on extraction failures

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies rotary position embeddings with partial-RoPE dimension slicing.

`inject_factorised_matrix_state(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands grouped key/value heads to match query heads.

Classes
-------

`FactorisedMatrixState(config: Any | None = None, rank: int | None = 32, d_model: int | Any | None = None, num_heads: int | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, old_attn: torch.nn.modules.module.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `W_q: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `W_v: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `c_proj: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `dense: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `k_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `key: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `o_proj: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `out_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `out_proj: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `q_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `query: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `v_lin: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    `value: neuralops.attentions.factorised_matrix_state.LoRALinear`
    :

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Transfer pretrained projection weights with strict Frobenius norm verification.

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear projection layer with frozen base weights and trainable residual LoRA adapters.
    
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

    `freeze_base_weights(self) ‑> None`
    :   Freeze base projection parameters so only LoRA parameters adapt.

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :   Unfreeze base projection parameters.