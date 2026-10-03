Module neuralops.compression_engine.ast_pruner
==============================================
AST: Adaptive Semi-Structured 2:4 and N:M Pruning with Empirical Fisher
Calibration and PCA Coordinate Rotation (Corrected).

Registered in COMPRESSOR_REGISTRY as 'ast_pruner' and 'ast'.

Mathematical Formulation (CORRECTED):
  1. Activation Covariance & PCA:
       Σ_X = (1/(N-1)) (X-μ)^T (X-μ) = Q Λ Q^T,  Q^T Q = I
  2. Empirical Jacobian-weighted saliency:
       g = ∂L/∂W,  L = 0.5·||logits||²,  F = E[g ⊙ g]
  3. PCA-rotated Fisher projected to original basis:
       F̃ = F · (Q ⊙ Q)             [in rotated basis]
       F_proj = F̃ · Q^T            [projected back to original basis]
  4. Saliency in ORIGINAL basis:
       S_ij = |W_ij| · sqrt(F_proj_ij + ε)
  5. 2:4 structured mask on ORIGINAL weights:
       W_pruned = Π_{2:4}(W, S)

Functions
---------

`ast_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

`ASTPrunedLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer with frozen AST-pruned base weights and optional trainable
    LoRA adapters.
    
    The base weight is stored in original coordinate basis with a true 2:4
    (or N:M) sparsity pattern. Framework may pack it as sparse if desired.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.ast_pruner.ASTPrunedLoRALinear`
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

    `merge(self) ‑> None`
    :   Merge LoRA into base weight in place.

    `merge_and_unload(self) ‑> None`
    :

    `unmerge(self) ‑> None`
    :

`ASTPruner(model: nn.Module, block_size: int = 4, n_keep: int = 2, dim: int = 1, use_pca: bool = True, use_fisher: bool = True, recalibrate_ln: bool = True, max_error: float = 0.2, fallback_to_original: bool = True, protect_critical: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, target_attention_only: bool = False, target_modules: list[str] | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, cfg: Any | None = None, **kwargs: Any)`
:   AST Pruner: vectorized semi-structured 2:4 / N:M pruning with empirical
    Jacobian calibration and PCA coordinate rotation (saliency only).
    
    Design notes:
      * PCA rotation is used **for saliency estimation only**. The actual
        2:4 mask is applied to the ORIGINAL weights so the sparsity pattern
        is preserved and remains exploitable by sparse backends.
      * The Jacobian proxy (0.5·||logits||²) is a heuristic inspired by
        SparseGPT; it is NOT the true Fisher Information Matrix.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor