Module neuralops.attentions.svd_linear_decay
============================================
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.

Functions
---------

`inject_svd_linear_decay(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

`inject_svd_linear_decay_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: torch.nn.modules.module.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`DecayGatedLinearAttention(*args, **kwargs)`
:   Decay Gated SVD Linear Attention.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.svd_linear_attention.SVDLinearAttention
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

`SVDLinear(in_features: int, out_features: int, rank: int, bias: bool = True, freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, use_fused_kernel: bool = False, use_rms_norm: bool = False, eps: float = 1e-06, dtype: torch.dtype = torch.float32, device: torch.device | None = None)`
:   Low-rank factorized linear layer:
        y = (x V^T) U^T + b + (alpha / r_lora) (x A^T) B^T
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :

    `get_weight(self) ‑> torch.Tensor`
    :

    `reset_parameters(self) ‑> None`
    :

    `set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) ‑> None`
    :

`SVDLinearAttention(config: Any | None = None, old_attn: nn.Module | None = None, rank: int | None = None, mode: str = 'exact', freeze_svd: bool = False, lora_rank: int | None = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, use_decay_gating: bool = False, decay_activation: str = 'softplus', decay_channel_wise: bool = False, use_fused_kernel: bool = False, use_rms_norm: bool = False, chunk_size: int = 64, **kwargs: Any)`
:   SVD Factorized Linear Attention with optional exponential decay gating.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.projection_factorized_attn.ProjectionFactorizedAttention
    * neuralops.attentions.svd_linear_decay.DecayGatedLinearAttention

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_bidirectional: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    `triton_kernel: str`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: nn.Module) ‑> None`
    :   SVDLA-FIX-01: never raise. On missing projections, log a warning
        and fall back to the randomly-initialized state.

    `get_orthogonality_loss(self) ‑> torch.Tensor`
    :

    `step_recurrent(self, x_t: torch.Tensor, past_state: tuple[torch.Tensor, torch.Tensor] | None = None) ‑> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]`
    :