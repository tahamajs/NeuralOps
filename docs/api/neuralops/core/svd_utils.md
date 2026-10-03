Module neuralops.core.svd_utils
===============================
Centralized SVD & Factorization Utilities for Attention and Compression Modules.
Provides numerical stability (FP32 execution for half-precision inputs),
truncated SVD factor initialization, and energy-based rank selection.

Functions
---------

`init_factorized_linear(src_weight: torch.Tensor, lin1: nn.Linear, lin2: nn.Linear, rank: int, src_bias: torch.Tensor | None = None) ‑> None`
:   Initializes a two-stage linear projection (x @ lin1.T @ lin2.T) from a dense source weight
    using truncated SVD:
        lin1.weight = Vh[:r, :]
        lin2.weight = U[:, :r] @ diag(S[:r])
    Ensures zeroed parameters for excess capacity and copies bias to lin2 if present.

`select_rank_by_energy(singular_values: torch.Tensor, energy_threshold: float = 0.9, min_rank: int = 4, max_rank: int | None = None) ‑> int`
:   Selects truncation rank r such that sum(S[:r]^2) / sum(S^2) >= energy_threshold.

`stable_svd(weight: torch.Tensor, full_matrices: bool = False) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
:   Computes SVD with guaranteed FP32 precision to prevent CUDA half-precision solver crashes (e.g. geqrf_cuda).
    Returns (U, S, Vh) matching the original device and floating dtype.