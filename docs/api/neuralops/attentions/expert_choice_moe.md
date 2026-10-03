Module neuralops.attentions.expert_choice_moe
=============================================
Expert Choice Mixture of Experts (MoE) Attention Block.

In Expert Choice routing (Zhou et al., 2022), experts select their top-k tokens
rather than tokens selecting top-k experts. This provides guaranteed perfect load balancing
across all experts, dynamic per-token expert allocation, and superior zero-shot transfer.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models
- Nested encoder detection to bypass out_lin in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- Step-0 representation preservation (zero-initialized expert outputs & biased gating)
- Numerically stable FP32 scores with nan_to_num protection
- Safe injection factory returning None on extraction failures
- Full cross-attention support

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies rotary position embeddings with partial-RoPE dimension slicing.

`inject_expert_choice_moe(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands grouped key/value heads to match query heads.

Classes
-------

`ExpertChoiceMoE(d_model: int, n_experts: int = 2, capacity_factor: float = 1.0, noise_std: float = 0.05)`
:   Expert Choice Routing MoE layer.
    Each expert selects top-k tokens with the highest affinity scores.
    Equipped with zero-initialized final layers for step-0 representation preservation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward execution using Expert Choice routing.

`ExpertChoiceMoEAttention(config: Any | None = None, rank: int | None = 64, n_experts: int = 2, capacity_factor: float = 1.0, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, moe_init_gate: float = 0.0, **kwargs: Any)`
:   Expert Choice MoE Multi-Head Attention Module.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
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

    `k_proj: LoRALinear`
    :

    `key: LoRALinear`
    :

    `o_proj: LoRALinear`
    :

    `out_proj: LoRALinear`
    :

    `q_proj: LoRALinear`
    :

    `query: LoRALinear`
    :

    `v_proj: LoRALinear`
    :

    `value: LoRALinear`
    :

    ### Methods

    `copy_weights(self, pretrained_attn: nn.Module) ‑> None`
    :   Copy projection weights from pretrained attention module with Frobenius norm verification.

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