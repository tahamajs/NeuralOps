Module neuralops.attentions.block_transformer_hiba
==================================================
Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).

Production-hardened implementation supporting:
1. Robust Pretrained Weight Extraction across DistilBERT, RoBERTa/BERT nested encoders,
   GPT-2 (fused Conv1D), and Qwen/LLaMA/Mistral.
2. Nested Encoder Detection to prevent double output projection signal attenuation in RoBERTa/BERT.
3. Native Grouped-Query Attention (GQA) support with kv_repeat broadcasting.
4. Rotary Position Embedding (RoPE) support with partial rotary dimension coordinate slicing.
5. Strict Causal Masking in all attention paths (intra-block, inter-block, and rectification) for decoders.
6. Parameter-Efficient Recovery Fine-Tuning with trainable residual LoRA adapters (lora_A, lora_B).
7. Cross-Attention support for encoder-decoder architectures.
8. Numerical stability via FP32 attention logits, nan_to_num guardrails, and -inf masking.
9. Static, stable layer indexing across forward passes.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies rotary position embeddings to query and key tensors with partial-RoPE dimension slicing.
    
    Supports coordinate layouts where head_dim > rotary_dim (e.g., Qwen3.5: head_dim=256, rotary_dim=64).
    Rotates the first `rotary_dim` coordinates and concatenates unrotated pass-through coordinates.

`inject_block_transformer_hiba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`BlockTransformerHIBA(config: Any | None = None, rank: int = 64, block_size: int = 64, top_k_inter: int = 4, rectification_interval: int = 4, layer_id: int | None = None, layer_idx: int | None = None, old_attn: torch.nn.modules.module.Module | None = None, pretrained_attn: torch.nn.modules.module.Module | None = None, comp_config: Any | None = None, **kwargs: Any)`
:   Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).
    
    Hardened for production use across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA/Mistral.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `c_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `dense: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `k_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `k_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `key: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `o_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `out_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `q_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `q_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `query: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `v_lin: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `v_proj: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    `value: neuralops.attentions.block_transformer_hiba.LoRALinear`
    :

    ### Methods

    `copy_weights(self, pretrained_attn: torch.nn.modules.module.Module) ‑> None`
    :   Extracts and copies pretrained weights across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA.
        Strictly validates that copied weights are non-empty and verifies reconstruction norms.

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear projection layer equipped with frozen base weights and trainable residual LoRA adapters.
    
    Base path:
        y_base = x @ W^T + b  (frozen when freeze_base=True)
    LoRA path:
        y_lora = (x @ lora_A^T) @ lora_B^T * scaling  (trainable)
    Total output:
        y = y_base + y_lora
    
    Uses standard `lora_A` and `lora_B` parameter naming recognized by framework recovery optimizers.
    
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