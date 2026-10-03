Module neuralops.compression_engine.fwsvd_compressor
====================================================
Fisher-Weighted Singular Value Decomposition (FWSVD) Compressor.

Computes diagonal Fisher information matrix F_i = E[x_i^2] from calibration activations,
weights the weight matrix W by diag(sqrt(F_i)), computes truncated SVD,
and unweights factors by diag(1 / sqrt(F_i)).
Uses the same structural SVDLinear replacement and fallback error logic as AASVD.

Classes
-------

`FWSVDCompressor(model: nn.Module, tokenizer: Any | None = None, rank: int = 64, max_error: float = 0.5, fallback_to_original: bool = True, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, target_attention_only: bool = True, **kwargs: Any)`
:   Fisher-Weighted SVD Compressor.
    Projects weights using diagonal Fisher activation weights F_i = E[x_i^2].
    Replaces dense nn.Linear with SVDLinear.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress_linear(self, layer: nn.Module, rank: int | None = None, fisher_diag: torch.Tensor | None = None) ‑> tuple[neuralops.compression_engine.aa_svd_compressor.SVDLinear | None, float, int]`
    :