Module neuralops.attentions.granite_hybrid
==========================================
Granite Hybrid Attention-SSM Layer with MoE Feed-Forward.

Combines Mamba-2 SSM recurrent layers with dense Multihead Attention (MHA) and
Top-k Mixture-of-Experts (MoE) feed-forward, with LoRA adapters and pretrained weight transfer.
Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_granite_hybrid(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, lora_rank: int = 16, **kwargs: Any) ‑> Any | None`
:   Injection hook for GraniteHybridBlock supporting single layers and whole models.

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expands grouped key/value heads to match query heads.

Classes
-------

`GraniteHybridBlock(config: Any | None = None, rank: int | None = None, dim: int | None = None, num_heads: int | None = None, num_experts: int = 4, top_k: int = 2, ratio: int = 2, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, **kwargs: Any)`
:   Granite Hybrid Attention-SSM Layer with MoE Feed-Forward and LoRA Adaptation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`Mamba2(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs: Any)`
:   Simplified Mamba-2 State Space Model (SSM) block.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`MixtureOfExperts(dim: int, ff_dim: int, num_experts: int = 4, top_k: int = 2, **kwargs: Any)`
:   Top-$k$ Token-Choice Mixture-of-Experts (MoE) Layer.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, **kwargs: Any) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.