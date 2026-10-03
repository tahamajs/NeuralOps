Module neuralops.compression_engine.drone_compressor
====================================================
DRONE Compressor: Data-aware Low-Rank SVD with Identity Weighting.

Implements truncated SVD directly on target linear weight matrices (identity weighting)
and replaces target layers with structural SVDLinear modules.
Applies the identical rank allocation and Frobenius reconstruction error fallback threshold as AASVD/FWSVD.

Classes
-------

`DRONECompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 64, max_error: float = 0.5, fallback_to_original: bool = True, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = True, **kwargs: Any)`
:   DRONE Compressor.
    Data-aware SVD with identity weighting (standard data-independent SVD baseline)
    using identical rank adaptation, parameter constraints, and fallback error mechanism as AASVD.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :