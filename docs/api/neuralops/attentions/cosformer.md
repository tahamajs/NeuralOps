Module neuralops.attentions.cosformer
=====================================
Cosformer: Rethinking Softmax in Attention with Linear Complexity.

Implements non-negative kernel feature mapping with cosine/sine positional reweighting,
cumulative sum causal linear attention for decoders, and associative accumulation for encoders.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models (vectorized prefix sum)
- Nested encoder detection to bypass out_proj in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- Complete value masking (masking both K and V, including last-chunk V masking)
- Numerically stable FP32 accumulation with nan_to_num protection
- Safe injection factory returning None on extraction failures
- Full cross-attention support
- Thread-safe injection depth guard
- Even head_dim validation for RoPE
- Transparent rank-agnostic behavior (Cosformer does not use rank)

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies rotary position embeddings with partial-RoPE dimension slicing.

`inject_cosformer(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any) ‑> Any | None`
:   Safe injection factory for CosformerAttention.
    
    Fix 3: Thread-safe recursion depth guard.
    Fix 13: pop pretrained_attn from kwargs to avoid conflict.
    Fix 14: Removed dead string expression.
    Fix 16: Clean kwargs before recursive calls.

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands grouped key/value heads to match query heads.

Classes
-------

`CosformerAttention(config: Any | None = None, rank: int | None = None, max_seq_len: int = 512, act_fun: str = 'relu', eps: float = 1e-06, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   Cosformer Attention with Linear Complexity and Exact Multi-Architecture Support.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel`
    :   The type of the None singleton.

    ### Instance variables

    `W_k: LoRALinear`
    :

    `W_q: LoRALinear`
    :

    `W_v: LoRALinear`
    :

    `c_proj: LoRALinear`
    :

    `dense: LoRALinear`
    :

    `k_lin: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_lin: LoRALinear`
    :

    `q_lin: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_lin: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, src: Any) ‑> None`
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