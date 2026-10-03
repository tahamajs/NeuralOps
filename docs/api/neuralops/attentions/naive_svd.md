Module neuralops.attentions.naive_svd
=====================================
Naive SVD baseline: plain truncated SVD without calibration, Fisher weighting, or whitening.
Used as an uncalibrated low-rank baseline in comparative evaluations.

Paper comparison reference:
Standard truncated SVD approximation W ~ U[:, :r] @ diag(S[:r]) @ V^T[:r, :].

Functions
---------

`inject_naive_svd(layer: nn.Module, comp_config: Any = None, old_attn: nn.Module | None = None, model_config: Any | None = None) ‑> torch.nn.modules.module.Module | None`
:   Factory hook to instantiate and inject NaiveSVDAttention into transformer layers.

Classes
-------

`NaiveSVDAttention(config: Any | None = None, model_config: Any | None = None, energy: float = 1.0, dropout: float = 0.1, pretrained_attn: nn.Module | None = None, rank: int | None = None, lora_rank: int = 0, lora_alpha: float = 1.0, is_decoder: bool | None = None, **kwargs: Any)`
:   Naive truncated SVD baseline attention module.
    Inherits production-hardened multi-architecture routing from MinimalSVDAttention
    with uncalibrated plain truncated SVD initialization and no recovery adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.attentions.minimal_svd.MinimalSVDAttention
    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module