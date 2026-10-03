Module neuralops.attentions.routing_attention
=============================================
Routing Attention: Content-Based Clustering for Sparse Attention – Production-Hardened.

Partitions sequence tokens into discrete clusters, confining attention computation
to intra-cluster tokens without materializing full [B, H, L, L] tensors.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_routing(model: nn.Module | None = None, layer: nn.Module | None = None, model_config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`inject_routing_attention(model: nn.Module | None = None, layer: nn.Module | None = None, model_config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`inject_routing_into_distilbert(model: nn.Module | None = None, layer: nn.Module | None = None, model_config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
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

`RoutingAttention(config: Any = None, rank: int | None = None, num_clusters: int = 16, n_iters: int = 3, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `k_lin`
    :

    `key`
    :

    `out_lin`
    :

    `q_lin`
    :

    `query`
    :

    `v_lin`
    :

    `value`
    :