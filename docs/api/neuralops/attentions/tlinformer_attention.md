Module neuralops.attentions.tlinformer_attention
================================================
TLinFormer: Exact linear-time attention via low-rank sequence projections.
Reference: LinFormer (Wang et al., 2020).

Production-hardened for DistilBERT, RoBERTa, GPT-2, Qwen3.5.

Functions
---------

`inject_tlinformer(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

Classes
-------

`TLinFormerAttention(config: Any | None = None, rank: int | None = 64, pretrained_attn: nn.Module | None = None, **kwargs: Any)`
:   TLinFormer: linear-time attention with sequence-dimension low-rank projection.
    
    For an encoder with N tokens and rank r ≤ N:
        K' = E K,  V' = F V,  E ∈ R^{r×N},  F ∈ R^{r×N}
        scores = Q K'ᵀ / √d      # (B, H, N, r)
        O = softmax(scores) V'   # (B, H, N, d)
    
    Complexity O(N · r · d) instead of O(N² · d).
    
    For decoders (causal), we deliberately fall back to dense attention. A
    projected key aggregates information from all N positions, so a causal
    mask over the projected keys cannot enforce causality — the model would
    silently leak future information. Dense causal attention is correct.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `E`
    :

    `F`
    :

    `c_proj`
    :

    `dense`
    :

    `k_lin`
    :

    `key`
    :

    `o_proj`
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