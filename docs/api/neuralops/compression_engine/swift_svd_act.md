Module neuralops.compression_engine.swift_svd_act
=================================================
Swift-SVD Activation-Aware Compressor — Production-Hardened (v2).

Registered as 'swift_svd_act', 'swift_svd', 'swift_svd_compressor'.

Algorithm:
  1. Collect empirical cross-covariance C = E[Y^T X] per layer across
     calibration data (online accumulation, bounded memory).
  2. Extract principal activation subspaces:
       U_act = top-r left  singular vectors of C
       V_act = top-r right singular vectors of C
  3. Project W onto (U_act, V_act):
       M = U_act^T @ W @ V_act
  4. SVD the projected matrix and reconstruct:
       A = U_act @ U_m · sqrt(S_m)
       B = sqrt(S_m) · Vt_m @ V_act^T
  5. Replace the dense layer with SVDLinear (or fall back to plain SVD).

Functions
---------

`swift_svd_act_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Swift-SVD activation-aware factory function.

Classes
-------

`SwiftSVDActivationCompressor(model: nn.Module, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, rank: int = 64, max_error: float = 0.5, fallback_to_original: bool = True, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, num_calibration_batches: int = 8, max_calibration_tokens: int = 32768, use_fp64_cross_cov: bool = True, **kwargs: Any)`
:   Swift-SVD Activation-Aware Compressor.
    
    Captures empirical input-output cross-covariances across calibration data,
    derives activation-aligned subspaces, projects weights onto the principal
    activation subspace, and replaces dense nn.Linear / Conv1D layers with
    factorized SVDLinear modules.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `compress_linear(self, layer: nn.Module, act_pair: tuple[torch.Tensor, int] | None = None) ‑> neuralops.compression_engine.aa_svd_compressor.SVDLinear | None`
    :