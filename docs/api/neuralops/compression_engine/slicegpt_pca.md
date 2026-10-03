Module neuralops.compression_engine.slicegpt_pca
================================================
SliceGPT PCA Compressor: Principal Component Analysis based Dimensionality Slicing.
Registered in COMPRESSOR_REGISTRY as 'slicegpt_pca' and 'slicegpt_pca_factory'.

Mathematical Formulation:
1. Activation Sample Matrix X ∈ R^{N × d_in}
2. Empirical Covariance: Σ = (1/(N-1)) (X - μ)^T (X - μ)
3. Eigendecomposition: Σ = V Λ V^T
4. Retain principal coordinates covering (1 - sparsity) cumulative variance:
   k = min{ r | Σ_{i≤r} λ_i / Σ_i λ_i ≥ 1 - sparsity }
5. Transform weights: W ← (W V) P_k P_k^T V^T
   Note: W_reconstructed = W @ V @ diag(1_k) @ V^T has rank ≤ k.
6. Optional LoRA Adapter Integration:
   y = (W_sliced x + b) + (α/r) (B A x)

=== REAL COMPRESSION (2026-09-25 FIX) ===
This version ACTUALLY reduces parameter count by replacing the sliced dense
weight with an SVDLinear low-rank factorization:

    W_reconstructed = W_pca[:, :keep] @ V[:, :keep].T
                    = U_factor @ Vt_factor     where:
                      U_factor  = W_pca[:, :keep]     shape (d_out, keep)
                      Vt_factor = V[:, :keep].T       shape (keep, d_in)

Parameter count drops from d_out * d_in to (d_out + d_in) * keep.

Functions
---------

`slicegpt_pca_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for SliceGPT PCA compressor.
    Returns (compressed_model, stats) if return_stats or return_tuple is True, else compressed_model.

Classes
-------

`SliceGPTPCACompressor(model: nn.Module, calibration_loader: Any | None = None, sparsity: float = 0.3, tokenizer: Any | None = None, calibration_texts: list[str] | None = None, skip_attention_qkv: bool = False, target_attention_only: bool = False, use_lora: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, max_error: float = 0.5, fallback_to_original: bool = True, min_numel: int = 16, target_modules: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Production-hardened SliceGPT PCA-based compressor that performs activation-aware
    principal coordinate projection and dimension slicing on Linear and Conv1D layers.
    
    Replaces each sliced dense weight with a real SVDLinear low-rank factorization
    so that parameter count actually drops from d_out*d_in to (d_out+d_in)*keep.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compresses the model using PCA-based principal subspace projection and
        real low-rank factorization (SVDLinear).
        
        Returns (compressed_model, stats).

`SliceGPTPCALoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer containing sliced/PCA-projected base weights (frozen)
    with trainable LoRA adapters (lora_A, lora_B) for post-compression fine-tuning.
    Base weights are frozen with requires_grad=False; initial LoRA contribution is zero.
    
    NOTE: This wrapper stores the base weight densely. For real parameter
    reduction, the compressor now uses SVDLinear instead. This class is kept
    for backward compatibility only.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, sliced_weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.slicegpt_pca.SliceGPTPCALoRALinear`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.