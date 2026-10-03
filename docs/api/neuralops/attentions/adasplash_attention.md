Module neuralops.attentions.adasplash_attention
===============================================
AdaSplash Attention: Adaptive Splash Attention with α-Entmax Sparse Activations.

Combines localized sliding-window splash radius with sparse α-entmax adaptive thresholding
and global token routing, ensuring robust numerical convergence and exact pretrained weight transfer.

[OPTIMIZED FOR ZERO-SHOT BENCHMARKING]

Functions
---------

`entmax15_bisect(scores: torch.Tensor, valid_mask: torch.Tensor | None = None, dim: int = -1, n_iter: int = 15) ‑> torch.Tensor`
:   Backward-compatible wrapper for 1.5-entmax.

`entmax_bisect(scores: torch.Tensor, alpha: float = 1.5, valid_mask: torch.Tensor | None = None, dim: int = -1, n_iter: int = 15) ‑> torch.Tensor`
:   Compute α-entmax via stabilized bisection strictly over valid tokens.
    Includes an Early Stopping mechanism to drastically improve Inference Speed.

`inject_adasplash(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any) ‑> neuralops.attentions.adasplash_attention.AdaSplashAttention | torch.nn.modules.module.Module | None`
:   

Classes
-------

`AdaSplashAttention(config: Any, rank: int | None = None, alpha: float = 1.1, n_iter: int = 15, splash_radius: int = 128, pretrained_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   AdaSplash Attention Layer with α-Entmax and Local Splash Radius.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `supports_rope: bool`
    :   The type of the None singleton.

    ### Methods

    `compute_bias(self, query_length: int, key_length: int, device: torch.device | None = None) ‑> torch.Tensor`
    :

`AdaSplashLinear(in_features: int, out_features: int, lora_rank: int = 16, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear projection layer with frozen base weights and trainable residual LoRA adapter.
    
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

    `reset_parameters(self) ‑> None`
    :