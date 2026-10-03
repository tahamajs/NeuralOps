Module neuralops.attentions.samba_moe
=====================================
Samba-MoE: Hybrid Selective State Space (SAMBA-M3) + REAP-MoE Block.
Combines local sliding-window attention, spectral Mamba SSM, and MoE routing.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Functions
---------

`inject_samba_moe(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.samba_moe.SambaMoEBlock | None`
:   Safe injection hook for SambaMoEBlock.

Classes
-------

`MinimalMoEFallback(d_model: int = 768, n_experts: int = 4)`
:   Fallback MoE layer when REAPMoE cannot be instantiated.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`SambaMoEBlock(config: Any | None = None, d_model: int | None = None, n_heads: int | None = None, n_experts: int = 8, window: int = 128, rank: int | None = 64, pretrained_attn: torch.nn.modules.module.Module | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, **kwargs: Any)`
:   SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Instance variables

    `is_nested_encoder: bool`
    :

    `key: torch.nn.modules.module.Module`
    :

    `query: torch.nn.modules.module.Module`
    :

    `value: torch.nn.modules.module.Module`
    :

    ### Methods

    `copy_weights(self, pretrained_attn: torch.nn.modules.module.Module) ‑> None`
    :   Propagate weight transfer to internal SAMBA-M3 and SSM components.