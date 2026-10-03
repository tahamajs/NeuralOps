Module neuralops.compression_engine.slicegpt
============================================
SliceGPT: Compress Large Language Models by Deleting Rows and Columns.
Registered in COMPRESSOR_REGISTRY as 'slicegpt' and 'slicegpt_compressor'.

Mathematical Formulation:
1. Orthogonal Coordinate Invariance:
   W_l X_l = (W_l Q_l) (Q_l^T X_l)
2. Eigen-decomposition of Activation Covariance:
   C = X X^T = Q Λ Q^T
3. Coordinate Slicing Transformation:
   W_sliced = W Q P_r,   P_r = [I_r, 0]^T

=== REAL COMPRESSION (2026-09-25 FIX) ===
This version ACTUALLY reduces parameter count by replacing the sliced dense
weight with an SVDLinear low-rank factorization:

    W_reconstructed = W_pca[:, :keep] @ V[:, :keep].T
                    = U_factor @ Vt_factor     where:
                      U_factor  = W_pca[:, :keep]     shape (d_out, keep)
                      Vt_factor = V[:, :keep].T       shape (keep, d_in)

The resulting parameter count is (d_out + d_in) * keep instead of
d_out * d_in, producing real compression when keep < d_out * d_in / (d_out + d_in).

Functions
---------

`slicegpt_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> Any`
:   Factory function for SliceGPT compressor.

Classes
-------

`SliceGPTCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.6, non_uniform: bool = False, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, max_error: float = 0.5, fallback_to_original: bool = True, target_modules: list[str] | None = None, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, **kwargs: Any)`
:   SliceGPT Compression Strategy (real low-rank implementation).
    
    Performs orthogonal rotation of each target weight matrix using the
    eigendecomposition of its input activation covariance, then truncates
    to the top-r principal coordinates. The truncated weight is stored as a
    rank-r SVDLinear factorization, achieving genuine parameter reduction.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Applies SliceGPT rotation + rank-r truncation and replaces each weight
        with a real SVDLinear low-rank factorization.