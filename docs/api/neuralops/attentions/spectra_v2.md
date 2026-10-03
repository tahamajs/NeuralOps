Module neuralops.attentions.spectra_v2
======================================
MHA-SDPA: A thin Multi-Head Attention wrapper around PyTorch's
scaled_dot_product_attention.

DISCLAIMER:
    This module was previously named "SPECTRA-v2" and claimed to
    implement a sparse per-head efficient context transformer
    with intra-block attention, top-k inter-block routing,
    adaptive spans, and RFF rectification. NONE of these
    features are implemented here. What this module actually
    does is:

      1. Project Q, K, V via LoRALinear (frozen base + adapters).
      2. Apply RoPE if position_embeddings are provided.
      3. Repeat KV heads for GQA.
      4. Call scaled_dot_product_attention.
      5. Apply head_mask.
      6. Project the output.

    This is functionally equivalent to a standard MHA with SDPA.
    It provides NO sparsity, NO top-k, NO adaptive span, and NO
    RFF features.

If you need the actual SPECTRA architecture, see paper/_SPECTRA_V2_REPORT.md.

Registered as "mha_sdpa" (canonical) and "spectra_v2" (deprecated).

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_spectra_v2(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, target_layer_idx: int | None = None, **kwargs: Any) ‑> Any | None`
:   Safely replace attention in model or layer with SpectraV2Attention (mha_sdpa).

Classes
-------

`LoRALinear(in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer augmented with Low-Rank Adaptation (LoRA) and base freezing.
    
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

`SpectraV2Attention(config: Any | None = None, model_config: Any | None = None, layer_idx: int = 0, total_layers: int = 12, bert_mode: bool = False, pretrained_attn: nn.Module | None = None, rank: int | None = 64, **kwargs: Any)`
:   MHA wrapper around PyTorch's scaled_dot_product_attention.
    Previously named spectra_v2 (deprecated alias).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Static methods

    `from_distilbert_pretrained(old_attn: nn.Module, model_config: Any | None = None, **kwargs: Any) ‑> neuralops.attentions.spectra_v2.SpectraV2Attention`
    :