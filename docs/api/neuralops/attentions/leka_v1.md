Module neuralops.attentions.leka_v1
===================================
LEKAv1: Linear-time Efficient Kernel Attention (Corrected with Positive Features).

Approximates the attention kernel via Positive Feature Mapping (ELU + 1)
to guarantee strict positivity and stable gradients, avoiding NaN denominator collapse.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights (FactorizedLoRALinear)
- GQA with repeat_kv
- RoPE position embeddings
- Causal chunked recurrence for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_leka_v1(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`FactorizedLoRALinear(in_features: int, out_features: int, rank: int = 64, lora_rank: int = 16, lora_alpha: float = 32.0, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   SVD-factorized linear + LoRA adapter.
    
    Forward: y = up(down(x)) + (lora_B @ lora_A)(x) * scaling
    Base (down, up) frozen; LoRA trainable.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

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

`LEKAV1Attention(config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 6, rank: int | None = None, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
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

    ### Methods

    `get_ortho_loss(self) ‑> float`
    :

    `kernel_feature_map(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Positive kernel feature embedding using ELU + 1.
        Guarantees strict positivity so denominator never collapses or becomes negative.
        Returns tensor of shape (B, H, N, F).