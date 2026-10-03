Module neuralops.attentions.res_svd
===================================
Spectral Residual Low-Rank SVD Attention (Res-SVD Attention) – Production-Hardened.

W = U_base V_base^T + alpha * B A
where B A is initialized from the top-k residual singular directions of W - W_base.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   

`inject_res_svd(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, target_layer_idx: int | None = None, **kwargs: Any)`
:   

`inject_res_svd_attention(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, target_layer_idx: int | None = None, **kwargs: Any)`
:   

Classes
-------

`ResSVDAttention(config: Any, base_rank: int | None = None, residual_rank: int | None = None, rank: int | None = 16, scaling: float = 1.0, factorize_base: bool = True, freeze_base: bool = True, dropout: float | None = None, pretrained_attn: nn.Module | None = None, is_decoder: bool | None = None, **kwargs: Any)`
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

    `k_proj`
    :

    `out_lin`
    :

    `q_lin`
    :

    `q_proj`
    :

    `v_lin`
    :

    `v_proj`
    :