Module neuralops.attentions.flash_attention3_fp8
================================================
FlashAttention-3 FP8: Low-Precision 8-bit Float Transformer Attention.

Implements FlashAttention-3 utilizing FP8 (e4m3fn) precision format for NVIDIA Hopper (SM 9.0+)
Tensor Cores with per-tensor dynamic scaling and de-scaling, with automatic fallback to
hardware-accelerated SDPA on pre-Hopper architectures (such as Ada Lovelace SM 8.9 or Ampere SM 8.0)
and CPU.

NOTE ON METHODOLOGY:
FlashAttention-3 FP8 is a hardware execution acceleration technique, NOT a model compression technique.
It does not reduce parameter count or weight storage (unless LoRA adapters are configured). Base model
weights are maintained in native precision and activation tensors are dynamically scaled and cast to FP8
during Hopper GPU attention computation.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Strict validation requiring all mandatory Q/K/V projections (raises RuntimeError if missing)
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models across FlashAttention and SDPA paths
- Nested encoder detection to bypass out_proj in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- Safe FP8 casting with per-tensor dynamic scaling (max absolute value / 448.0) and de-scaling
- Hardware capability gating (Hopper compute capability >= 9.0 check)
- Full cross-attention support

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies rotary position embeddings with partial-RoPE dimension slicing.

`inject_flash3_fp8(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands grouped key/value heads to match query heads.

Classes
-------

`FlashAttention3FP8(config: Any | None = None, rank: int | None = None, d_model: int | Any = 768, n_heads: int = 12, use_fp8: bool = False, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, layer_id: int | None = None, layer_idx: int | None = None, **kwargs: Any)`
:   FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.
    
    Accelerates attention using FP8 (e4m3fn) precision with FlashAttention-3 kernels on
    Hopper (SM 9.0+) GPUs. Automatically falls back to native PyTorch SDPA on pre-Hopper
    hardware or CPU.
    
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