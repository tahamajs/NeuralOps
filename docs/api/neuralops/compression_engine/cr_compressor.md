Module neuralops.compression_engine.cr_compressor
=================================================
CR (Column-Row / Interpolative Decomposition) Compressor.
Decomposes W ≈ C @ P, where C are actual columns of W.
Registered in COMPRESSOR_REGISTRY as 'cr' and 'cr_compressor'.

Functions
---------

`cr_compressor_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for CR compressor.

Classes
-------

`CRCompressor(model: nn.Module | None = None, rank: int = 64, max_error: float = 0.2, fallback_to_original: bool = True, max_retries: int = 2, target_attention_only: bool = False, cfg: Any | None = None, target_layers: list[str] | None = None, **kwargs: Any)`
:   CR (Column-Row / Interpolative Decomposition) Compressor.
    Decomposes linear layers into skeleton column bases and interpolative projections.
    
    Args:
        model: PyTorch model.
        rank: Rank parameter (number of skeleton columns).
        max_error: Maximum relative Frobenius reconstruction error (default 0.20).
        fallback_to_original: If True, keep original layer when error exceeds max_error (default True).
        max_retries: Number of rank increase retries before fallback (default 2).
        target_attention_only: If True, restrict compression to attention projections only.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using interpolative decomposition.

    `compress_linear(self, linear: nn.Linear | Any) ‑> neuralops.compression_engine.cr_compressor.CRLinear`
    :   Compress a single linear layer into CRLinear.

`CRLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Interpolative Decomposition / Column-Row Factorized Linear Layer.
    W ≈ C @ P where:
    - C in R^{out_features x rank}: column skeleton from W
    - P in R^{rank x in_features}: interpolative projection matrix
    
    Forward pass:
    y = ((x @ P^T) @ C^T) + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 64) ‑> neuralops.compression_engine.cr_compressor.CRLinear`
    :   Perform Column Interpolative Decomposition on linear or Conv1D layer weight matrix.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix (out_features, in_features).

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution:
        y = ((x @ P^T) @ C^T) + bias

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize full reconstructed weight matrix.