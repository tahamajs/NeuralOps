Module neuralops.attentions.performer_attention
===============================================
Performer Attention – Fast Attention via Positive Orthogonal Random Features (FAVOR+).

Production-hardened with full multi-architecture support.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings with source_attn fallback
- Exact causal FAVOR+ prefix scan for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_performer(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, **kwargs: Any) ‑> Any | None`
:   Injection hook for PerformerAttention supporting single layers and whole models.
    
    FIX 15: uses thread-local recursion depth.

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer with optional LoRA adapter.
    
    When lora_rank == 0, behaves exactly like nn.Linear.
    When lora_rank > 0, adds trainable LoRA branches A, B.
    
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
    :

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

`PerformerAttention(config: Any | None = None, rank: int | None = None, eps: float = 1e-06, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, is_nested_encoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, src: Any) ‑> bool`
    :   Copy weights from source attention module.
        
        FIX 13: returns bool (was None).
        FIX 4: sets is_nested_encoder robustly.