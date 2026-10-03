Module neuralops.compression_engine.cur_compressor
==================================================
CUR Compressor: replaces Linear and Conv1D layers with CUR decomposition.
Uses leverage score sampling or energy sampling for column/row selection.
Registered in COMPRESSOR_REGISTRY as 'cur' and 'cur_compressor'.

Functions
---------

`compute_energy_scores(W: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Compute Frobenius energy sampling probabilities for columns and rows.

`compute_leverage_scores(W: torch.Tensor, rank: int) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Compute exact statistical leverage scores from the top-k singular vectors of W.
    W: shape (out_features, in_features)
    Returns:
        (col_scores, row_scores) normalized to valid probability distributions summing to 1.0.

`cur_compressor_factory(model: nn.Module, comp_config: Any | None = None, rank: int | None = None, **kwargs: Any) ‑> torch.nn.modules.module.Module | tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
:   Factory function for CUR compression.

`select_indices(scores: torch.Tensor, k: int, deterministic: bool = True) ‑> torch.Tensor`
:   Select k indices based on scores (either deterministic top-k or multinomial sampling).

`tikhonov_pinv(A: torch.Tensor, reg: float = 0.001) ‑> torch.Tensor`
:   Computes Tikhonov regularized pseudo-inverse of matrix A: (A^T A + reg * I)^(-1) A^T
    or via SVD regularized inversion for numerical stability.

Classes
-------

`CURCompressor(model: nn.Module | None = None, rank: int = 64, min_rank: int = 8, method: str = 'leverage', deterministic: bool = True, max_error: float = 0.2, fallback_to_original: bool = True, max_retries: int = 2, lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False, target_attention_only: bool = False, target_names: list[str] | None = None, exclude_names: list[str] | None = None, cfg: Any | None = None, **kwargs: Any)`
:   Compressor that replaces eligible Linear and Conv1D layers with CURLinear.
    
    Args:
        model: PyTorch model.
        rank: Target rank for skeleton column and row selection (default 64).
        min_rank: Minimum allowable rank for compression (default 8).
        method: Selection method ('leverage' or 'energy', default 'leverage').
        deterministic: Deterministic top-k selection if True (default True).
        max_error: Maximum relative Frobenius reconstruction error allowed (default 0.20).
        fallback_to_original: If True, retains original layer when reconstruction error exceeds max_error (default True).
        max_retries: Number of adaptive rank retries when error threshold is exceeded (default 2).
        lora_r: LoRA adapter rank if adding fine-tuning adapters (default 0).
        lora_alpha: LoRA scaling factor (default 1.0).
        freeze_skeletons: If True, freeze C and R parameters during post-compression training (default False).
        target_attention_only: If True, restrict compression to attention projections only.
        target_names: Optional list of layer name substrings to target.
        exclude_names: Optional list of layer name substrings to exclude.
        cfg: Optional configuration object.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, model: nn.Module | None = None) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Replace eligible Linear and Conv1D layers with CURLinear.

    `compress_linear(self, linear: nn.Linear | Any) ‑> neuralops.compression_engine.cur_compressor.CURLinear`
    :   Compress a single linear layer into CURLinear.

`CURLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, method: str = 'leverage', lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear layer whose weight matrix is factorized via CUR cross-approximation:
    W ≈ C @ U @ R
    - C in R^{out_features x rank}: selected columns of W
    - U in R^{rank x rank}: Moore-Penrose pseudo-inverse core matrix
    - R in R^{rank x in_features}: selected rows of W
    
    Executes forward pass in factorized form:
    y = ((x @ R^T) @ U^T) @ C^T + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 64, method: str = 'leverage', deterministic: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False) ‑> neuralops.compression_engine.cur_compressor.CURLinear`
    :   Construct CURLinear from existing nn.Linear or Conv1D using leverage scores or energy sampling.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution:
        x: (..., in_features)
        y = ((x @ R^T) @ U^T) @ C^T + bias

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize reconstructed weight matrix W = C @ U @ R.

`CURLowRankLinear(in_features: int, out_features: int, rank: int = 64, bias: bool = True, method: str = 'leverage', lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False, dtype: torch.dtype | None = None, device: torch.device | None = None)`
:   Linear layer whose weight matrix is factorized via CUR cross-approximation:
    W ≈ C @ U @ R
    - C in R^{out_features x rank}: selected columns of W
    - U in R^{rank x rank}: Moore-Penrose pseudo-inverse core matrix
    - R in R^{rank x in_features}: selected rows of W
    
    Executes forward pass in factorized form:
    y = ((x @ R^T) @ U^T) @ C^T + bias
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Linear | Any, rank: int = 64, method: str = 'leverage', deterministic: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, freeze_skeletons: bool = False) ‑> neuralops.compression_engine.cur_compressor.CURLinear`
    :   Construct CURLinear from existing nn.Linear or Conv1D using leverage scores or energy sampling.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Sub-quadratic forward execution:
        x: (..., in_features)
        y = ((x @ R^T) @ U^T) @ C^T + bias

    `get_weight(self) ‑> torch.Tensor`
    :   Materialize reconstructed weight matrix W = C @ U @ R.