"""
Drift detector with a registry of statistical methods, Hutchinson randomized trace estimation,
Nyström low-rank approximations, and stable 2-Wasserstein Hessian metrics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats

from framework.core.registry import Registry
from .report import DriftReport, FeatureDriftReport

METHOD_REGISTRY = Registry("drift_methods")


# ── Matrix Square Root & Wasserstein-2 Stability Helpers ────────────


def matrix_square_root(
    matrix: np.ndarray,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
) -> np.ndarray:
    """
    Computes the stable symmetric positive semi-definite matrix square root A^{1/2}.

    Applies Tikhonov damping:
        A_reg = A + lambda * I
    and clamps negative/sub-threshold eigenvalues to min_eig to ensure strict
    positive semi-definiteness prior to square root calculation.

    Args:
        matrix: Input 2D square matrix (e.g. Hessian or Covariance).
        damping: Tikhonov regularization factor lambda > 0.
        min_eig: Minimum eigenvalue threshold for clamping (default 1e-8).

    Returns:
        np.ndarray: Matrix square root A^{1/2}.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    dim = matrix.shape[0]

    # RESEARCH FIX: Add Tikhonov damping (H = H + lambda * I) to prevent matrix ill-conditioning and singularity
    reg_matrix = matrix + damping * np.eye(dim, dtype=np.float64)

    # Symmetrize to prevent numerical asymmetry
    sym_matrix = 0.5 * (reg_matrix + reg_matrix.T)

    # Eigenvalue decomposition for symmetric/Hermitian matrices
    eigvals, eigvecs = np.linalg.eigh(sym_matrix)

    # RESEARCH FIX: Clamp negative and near-zero eigenvalues to min_eig before calculating matrix square roots
    eigvals_clamped = np.maximum(eigvals, min_eig)

    # Reconstruct matrix square root: A^{1/2} = V * diag(sqrt(lambda)) * V^T
    sqrt_matrix = eigvecs @ np.diag(np.sqrt(eigvals_clamped)) @ eigvecs.T
    return 0.5 * (sqrt_matrix + sqrt_matrix.T)


# ── Hutchinson's Randomized Trace Estimator ─────────────────────────


def hutchinson_trace_estimator(
    matrix_or_mv: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    dim: Optional[int] = None,
    num_samples: int = 50,
    distribution: str = "rademacher",
    seed: Optional[int] = None,
) -> float:
    """
    Computes Hutchinson's randomized trace estimator:
        Tr(A) = E_{v ~ D}[v^T A v] approx (1/m) sum_{j=1}^m v_j^T (A v_j)

    Eliminates O(d^3) dense matrix operations by computing stochastic trace in O(m * d) operations.

    Args:
        matrix_or_mv: Either a 2D square matrix (d x d) or a callable computing matrix-vector product A @ v.
        dim: Dimension d of the matrix (inferred if 2D array is provided).
        num_samples: Number of random probe vectors m (default: 50).
        distribution: Probe distribution, either 'rademacher' (+/- 1) or 'gaussian' (N(0, 1)).
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        float: Estimated trace Tr(A).
    """
    # RESEARCH FIX: Hutchinson randomized trace estimator for stochastic trace evaluation
    # with Rademacher or Gaussian probe vectors, avoiding full O(d^3) matrix eigendecompositions
    rng = np.random.RandomState(seed)

    if isinstance(matrix_or_mv, np.ndarray):
        matrix = np.asarray(matrix_or_mv, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square 2D matrix, got shape {matrix.shape}")
        d = matrix.shape[0]

        def mv_fn(v: np.ndarray) -> np.ndarray:
            return matrix @ v

    elif callable(matrix_or_mv):
        if dim is None or dim <= 0:
            raise ValueError("dim must be specified when matrix_or_mv is a callable function")
        d = dim
        mv_fn = matrix_or_mv
    else:
        raise TypeError("matrix_or_mv must be a 2D numpy array or a callable function")

    dist_lower = distribution.lower()
    if dist_lower == "rademacher":
        # Random +/- 1 Rademacher vectors
        probe_vectors = rng.choice([-1.0, 1.0], size=(d, num_samples)).astype(np.float64)
    elif dist_lower == "gaussian":
        # Standard normal N(0, 1) Gaussian vectors
        probe_vectors = rng.randn(d, num_samples).astype(np.float64)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'. Choose 'rademacher' or 'gaussian'.")

    trace_accum = 0.0
    for j in range(num_samples):
        v_j = probe_vectors[:, j]
        av_j = mv_fn(v_j)
        trace_accum += float(np.dot(v_j, av_j))

    return float(trace_accum / num_samples)


# ── Nyström Low-Rank Approximation & Fast Wasserstein ────────────────


def nystrom_low_rank_approximation(
    matrix: np.ndarray,
    rank: int = 20,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Nyström low-rank approximation of a symmetric positive semi-definite matrix:
        Sigma approx U_r Lambda_r U_r^T
    where U_r in R^{d x r} is orthonormal (U_r^T U_r = I_r) and Lambda_r in R^r are positive eigenvalues.

    Args:
        matrix: Input 2D symmetric PSD matrix Sigma in R^{d x d}.
        rank: Target low rank r << d.
        damping: Tikhonov regularization factor lambda.
        min_eig: Floor threshold for clamped eigenvalues.
        seed: Optional random seed for deterministic sketch matrix generation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (U_r, Lambda_r) where U_r is (d, r) and Lambda_r is (r,).
    """
    # RESEARCH FIX: Randomized Nystrom low-rank approximation reducing O(d^3) dense
    # eigendecomposition to O(d r^2 + r^3) complexity for scalable 2-Wasserstein computation
    matrix = np.asarray(matrix, dtype=np.float64)
    d = matrix.shape[0]
    effective_rank = min(rank, d)

    if d <= effective_rank:
        # Exact eigendecomposition when dimension is smaller than target rank
        reg_matrix = matrix + damping * np.eye(d, dtype=np.float64)
        sym_matrix = 0.5 * (reg_matrix + reg_matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym_matrix)
        eigvals_clamped = np.maximum(eigvals, min_eig)
        # Sort descending
        sort_idx = np.argsort(eigvals_clamped)[::-1]
        return eigvecs[:, sort_idx], eigvals_clamped[sort_idx]

    rng = np.random.RandomState(seed)
    # Gaussian test / sketch matrix Omega in R^{d x r}
    omega = rng.randn(d, effective_rank).astype(np.float64)

    # Form sample sketch Y = Sigma @ Omega in R^{d x r}
    y = matrix @ omega

    # Orthonormalize columns via QR: Q in R^{d x r}, Q^T Q = I_r
    q, _ = np.linalg.qr(y, mode="reduced")

    # Project matrix onto subspace: B = Q^T @ Sigma @ Q in R^{r x r}
    b = q.T @ (matrix @ q)
    b_reg = 0.5 * (b + b.T) + damping * np.eye(effective_rank, dtype=np.float64)

    # Small r x r symmetric eigendecomposition (costs O(r^3))
    eigvals_b, eigvecs_b = np.linalg.eigh(b_reg)
    eigvals_clamped = np.maximum(eigvals_b, min_eig)

    # Sort descending
    sort_idx = np.argsort(eigvals_clamped)[::-1]
    eigvals_sorted = eigvals_clamped[sort_idx]
    eigvecs_sorted = eigvecs_b[:, sort_idx]

    # Full orthonormal basis U_r = Q @ eigvecs_sorted in R^{d x r}
    u_r = q @ eigvecs_sorted

    return u_r, eigvals_sorted


def nystrom_matrix_square_root(
    matrix: np.ndarray,
    rank: int = 20,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
    return_factor: bool = False,
    seed: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the Nyström low-rank matrix square root Sigma^{1/2} approx U_r diag(sqrt(Lambda_r)) U_r^T
    or its square root factor F = U_r diag(sqrt(Lambda_r)) in R^{d x r}.

    Args:
        matrix: Input 2D symmetric PSD matrix in R^{d x d}.
        rank: Target rank r.
        damping: Regularization factor.
        min_eig: Minimum eigenvalue clamp floor.
        return_factor: If True, returns factor F in R^{d x r} where Sigma approx F @ F^T.
        seed: Random seed.

    Returns:
        np.ndarray: Reconstructed square root matrix (or factor F if return_factor=True).
    """
    # RESEARCH FIX: Low-rank matrix square root construction via Nystrom factors
    u_r, lambda_r = nystrom_low_rank_approximation(
        matrix, rank=rank, damping=damping, min_eig=min_eig, seed=seed
    )
    sqrt_lambda = np.sqrt(lambda_r)
    factor_f = u_r * sqrt_lambda[np.newaxis, :]  # (d, r)

    if return_factor:
        return factor_f

    # Full reconstruction Sigma^{1/2} = F @ U_r^T = U_r diag(sqrt(Lambda)) U_r^T
    sqrt_matrix = factor_f @ u_r.T
    return 0.5 * (sqrt_matrix + sqrt_matrix.T)


def fast_bures_wasserstein_nystrom(
    cov1: np.ndarray,
    cov2: np.ndarray,
    mean1: Optional[np.ndarray] = None,
    mean2: Optional[np.ndarray] = None,
    rank: int = 20,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
    seed: Optional[int] = None,
) -> float:
    """
    Computes the 2-Wasserstein (Bures-Wasserstein) metric in O(d r^2 + r^3) time
    using Nyström low-rank factor SVD nuclear norm:
        Tr((Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2}) = ||F_1^T F_2||_* = sum sigma_i(F_1^T F_2)
    where F_1 = U_1 Lambda_1^{1/2} and F_2 = U_2 Lambda_2^{1/2}.

    Args:
        cov1: Reference covariance or Hessian matrix Sigma_1 in R^{d x d}.
        cov2: Current covariance or Hessian matrix Sigma_2 in R^{d x d}.
        mean1: Optional mean vector mu_1 in R^d.
        mean2: Optional mean vector mu_2 in R^d.
        rank: Low-rank approximation rank r.
        damping: Tikhonov regularization.
        min_eig: Minimum eigenvalue floor.
        seed: Random seed.

    Returns:
        float: 2-Wasserstein distance.
    """
    # RESEARCH FIX: Closed-form low-rank 2-Wasserstein cross-term calculation via
    # nuclear norm of small (r x r) cross-projection matrix F_1^T F_2, bypassing full O(d^3) eigendecompositions
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    if cov1.ndim == 1:
        cov1 = np.diag(cov1)
    if cov2.ndim == 1:
        cov2 = np.diag(cov2)

    mean_term = 0.0
    if mean1 is not None and mean2 is not None:
        diff = np.asarray(mean1, dtype=np.float64) - np.asarray(mean2, dtype=np.float64)
        mean_term = float(np.sum(diff ** 2))

    # Obtain Nystrom factors: Sigma_1 approx F_1 F_1^T, Sigma_2 approx F_2 F_2^T
    u1, lambda1 = nystrom_low_rank_approximation(
        cov1, rank=rank, damping=damping, min_eig=min_eig, seed=seed
    )
    u2, lambda2 = nystrom_low_rank_approximation(
        cov2, rank=rank, damping=damping, min_eig=min_eig, seed=seed
    )

    f1 = u1 * np.sqrt(lambda1)[np.newaxis, :]  # (d, r1)
    f2 = u2 * np.sqrt(lambda2)[np.newaxis, :]  # (d, r2)

    # Compute small r1 x r2 cross-matrix C = F_1^T F_2
    c = f1.T @ f2  # (r1, r2)

    # Compute singular values of C (nuclear norm ||C||_*)
    s = np.linalg.svd(c, compute_uv=False)
    tr_inner = float(np.sum(s))

    # Low-rank or exact traces
    tr_cov1 = float(np.sum(lambda1))
    tr_cov2 = float(np.sum(lambda2))

    w2_sq = mean_term + tr_cov1 + tr_cov2 - 2.0 * tr_inner
    return float(np.sqrt(max(0.0, w2_sq)))


# RESEARCH FIX: Compatibility alias for fast Nystrom Wasserstein distance calculation
fast_wasserstein_distance_nystrom = fast_bures_wasserstein_nystrom


def bures_wasserstein_distance(
    cov1: np.ndarray,
    cov2: np.ndarray,
    mean1: Optional[np.ndarray] = None,
    mean2: Optional[np.ndarray] = None,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
    use_nystrom: bool = False,
    rank: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Computes the 2-Wasserstein (Bures-Wasserstein) metric between Gaussian distributions
    or second-order Hessian/covariance matrices:
        W_2^2(mu_1, Sigma_1; mu_2, Sigma_2) = ||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2 (Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2})

    Args:
        cov1: Reference covariance or Hessian matrix Sigma_1 in R^{d x d}.
        cov2: Current covariance or Hessian matrix Sigma_2 in R^{d x d}.
        mean1: Optional reference mean vector mu_1 in R^d.
        mean2: Optional current mean vector mu_2 in R^d.
        damping: Tikhonov regularization parameter lambda.
        min_eig: Eigenvalue clamp floor (1e-8).
        use_nystrom: If True, uses fast Nyström low-rank SVD nuclear norm calculation.
        rank: Low-rank rank r for Nyström approximation.
        seed: Random seed.

    Returns:
        float: 2-Wasserstein distance.
    """
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    if cov1.ndim == 1:
        cov1 = np.diag(cov1)
    if cov2.ndim == 1:
        cov2 = np.diag(cov2)

    dim = cov1.shape[0]
    if use_nystrom or (rank is not None and rank < dim):
        effective_rank = rank if rank is not None else min(20, dim)
        return fast_bures_wasserstein_nystrom(
            cov1,
            cov2,
            mean1=mean1,
            mean2=mean2,
            rank=effective_rank,
            damping=damping,
            min_eig=min_eig,
            seed=seed,
        )

    mean_term = 0.0
    if mean1 is not None and mean2 is not None:
        diff = np.asarray(mean1, dtype=np.float64) - np.asarray(mean2, dtype=np.float64)
        mean_term = float(np.sum(diff ** 2))

    # RESEARCH FIX: Compute square root of Sigma_1 with Tikhonov damping and eigenvalue clamping to 1e-8
    cov1_sqrt = matrix_square_root(cov1, damping=damping, min_eig=min_eig)

    # Inner product matrix M = Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2}
    inner_mat = cov1_sqrt @ cov2 @ cov1_sqrt

    # RESEARCH FIX: Compute square root of M with Tikhonov damping and eigenvalue clamping to 1e-8
    inner_sqrt = matrix_square_root(inner_mat, damping=damping, min_eig=min_eig)

    tr_cov1 = float(np.trace(cov1))
    tr_cov2 = float(np.trace(cov2))
    tr_inner = float(np.trace(inner_sqrt))

    w2_sq = mean_term + tr_cov1 + tr_cov2 - 2.0 * tr_inner
    return float(np.sqrt(max(0.0, w2_sq)))


# ── Method implementations ──────────────────────────────────────────


@METHOD_REGISTRY.register("ks")
def kolmogorov_smirnov(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Kolmogorov‑Smirnov test for continuous distributions."""
    stat, p = stats.ks_2samp(ref, cur)
    threshold = kwargs.get("threshold", 0.1)
    return float(stat), float(p) if p is not None else None, bool(stat > threshold)


@METHOD_REGISTRY.register("psi")
def population_stability_index(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Population Stability Index."""
    bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="auto")
    ref_hist, _ = np.histogram(ref, bins=bins, density=True)
    cur_hist, _ = np.histogram(cur, bins=bins, density=True)
    ref_hist = np.clip(ref_hist, 1e-10, None)
    cur_hist = np.clip(cur_hist, 1e-10, None)
    psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
    threshold = kwargs.get("threshold", 0.1)
    return float(psi), None, bool(psi > threshold)


@METHOD_REGISTRY.register("wasserstein")
def wasserstein(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """
    Wasserstein distance (Earth Mover's Distance for 1D, or 2-Wasserstein for 2D Hessian/Covariance).
    """
    threshold = kwargs.get("threshold", 0.1)
    damping = kwargs.get("damping", 1e-6)
    min_eig = kwargs.get("min_eig", 1e-8)

    if ref.ndim == 2 and cur.ndim == 2:
        if ref.shape[0] == ref.shape[1] and cur.shape[0] == cur.shape[1]:
            stat = bures_wasserstein_distance(ref, cur, damping=damping, min_eig=min_eig)
            return float(stat), None, bool(stat > threshold)
        elif ref.shape[1] == cur.shape[1]:
            mean1, mean2 = np.mean(ref, axis=0), np.mean(cur, axis=0)
            cov1, cov2 = np.cov(ref, rowvar=False), np.cov(cur, rowvar=False)
            stat = bures_wasserstein_distance(cov1, cov2, mean1=mean1, mean2=mean2, damping=damping, min_eig=min_eig)
            return float(stat), None, bool(stat > threshold)

    stat = stats.wasserstein_distance(ref.ravel(), cur.ravel())
    return float(stat), None, bool(stat > threshold)


@METHOD_REGISTRY.register("wasserstein_2")
@METHOD_REGISTRY.register("hessian_wasserstein")
def hessian_wasserstein(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """
    Second-order 2-Wasserstein distance for Hessian matrices with Tikhonov damping
    and eigenvalue clamping.
    """
    threshold = kwargs.get("threshold", 0.1)
    damping = kwargs.get("damping", 1e-6)
    min_eig = kwargs.get("min_eig", 1e-8)

    # If 1D data passed, convert to covariance / dispersion matrices
    if ref.ndim == 1 and cur.ndim == 1:
        ref_mat = np.array([[np.var(ref)]])
        cur_mat = np.array([[np.var(cur)]])
        mean1 = np.array([np.mean(ref)])
        mean2 = np.array([np.mean(cur)])
    elif ref.ndim == 2 and cur.ndim == 2 and ref.shape[0] != ref.shape[1]:
        # Sample data matrices (N x D)
        mean1 = np.mean(ref, axis=0)
        mean2 = np.mean(cur, axis=0)
        ref_mat = np.cov(ref, rowvar=False)
        cur_mat = np.cov(cur, rowvar=False)
    else:
        ref_mat = ref
        cur_mat = cur
        mean1 = None
        mean2 = None

    stat = bures_wasserstein_distance(
        ref_mat, cur_mat, mean1=mean1, mean2=mean2, damping=damping, min_eig=min_eig
    )
    return float(stat), None, bool(stat > threshold)


@METHOD_REGISTRY.register("wasserstein_2_nystrom")
@METHOD_REGISTRY.register("fast_wasserstein")
@METHOD_REGISTRY.register("nystrom_wasserstein")
def fast_wasserstein_detector(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """
    Fast 2-Wasserstein distance detector using Nyström low-rank matrix square root approximation.
    """
    threshold = kwargs.get("threshold", 0.1)
    damping = kwargs.get("damping", 1e-6)
    min_eig = kwargs.get("min_eig", 1e-8)
    rank = kwargs.get("rank", 20)
    seed = kwargs.get("seed", None)

    if ref.ndim == 1 and cur.ndim == 1:
        ref_mat = np.array([[np.var(ref)]])
        cur_mat = np.array([[np.var(cur)]])
        mean1 = np.array([np.mean(ref)])
        mean2 = np.array([np.mean(cur)])
    elif ref.ndim == 2 and cur.ndim == 2 and ref.shape[0] != ref.shape[1]:
        mean1 = np.mean(ref, axis=0)
        mean2 = np.mean(cur, axis=0)
        ref_mat = np.cov(ref, rowvar=False)
        cur_mat = np.cov(cur, rowvar=False)
    else:
        ref_mat = ref
        cur_mat = cur
        mean1 = None
        mean2 = None

    stat = fast_bures_wasserstein_nystrom(
        ref_mat,
        cur_mat,
        mean1=mean1,
        mean2=mean2,
        rank=rank,
        damping=damping,
        min_eig=min_eig,
        seed=seed,
    )
    return float(stat), None, bool(stat > threshold)


@METHOD_REGISTRY.register("hutchinson_trace")
def hutchinson_trace_detector(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Hutchinson randomized trace drift detector."""
    threshold = kwargs.get("threshold", 0.1)
    num_samples = kwargs.get("num_samples", 50)
    distribution = kwargs.get("distribution", "rademacher")
    seed = kwargs.get("seed", None)

    if ref.ndim == 2 and ref.shape[0] == ref.shape[1]:
        ref_tr = hutchinson_trace_estimator(
            ref, num_samples=num_samples, distribution=distribution, seed=seed
        )
        cur_tr = hutchinson_trace_estimator(
            cur, num_samples=num_samples, distribution=distribution, seed=seed
        )
    else:
        ref_cov = (
            np.cov(ref, rowvar=False)
            if (ref.ndim == 2 and ref.shape[0] != ref.shape[1])
            else np.diag(ref.ravel())
        )
        cur_cov = (
            np.cov(cur, rowvar=False)
            if (cur.ndim == 2 and cur.shape[0] != cur.shape[1])
            else np.diag(cur.ravel())
        )
        ref_tr = hutchinson_trace_estimator(
            ref_cov, num_samples=num_samples, distribution=distribution, seed=seed
        )
        cur_tr = hutchinson_trace_estimator(
            cur_cov, num_samples=num_samples, distribution=distribution, seed=seed
        )

    diff = abs(ref_tr - cur_tr) / (abs(ref_tr) + 1e-10)
    return float(diff), None, bool(diff > threshold)


@METHOD_REGISTRY.register("js")
def jensen_shannon(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Jensen‑Shannon divergence."""
    bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="auto")
    ref_hist, _ = np.histogram(ref, bins=bins, density=True)
    cur_hist, _ = np.histogram(cur, bins=bins, density=True)
    ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
    cur_hist = cur_hist / (cur_hist.sum() + 1e-10)
    m = (ref_hist + cur_hist) / 2
    kl_ref = np.sum(ref_hist * np.log(ref_hist / (m + 1e-10) + 1e-10))
    kl_cur = np.sum(cur_hist * np.log(cur_hist / (m + 1e-10) + 1e-10))
    js = (kl_ref + kl_cur) / 2
    threshold = kwargs.get("threshold", 0.1)
    return float(js), None, bool(js > threshold)


@METHOD_REGISTRY.register("chi2")
def chi_squared(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Chi‑squared test (for categorical / binned data)."""
    bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="auto")
    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)
    scale = ref.sum() / cur.sum() if cur.sum() > 0 else 1.0
    cur_counts = cur_counts * scale
    mask = (ref_counts > 0) | (cur_counts > 0)
    chi2, p = stats.chisquare(f_obs=cur_counts[mask], f_exp=ref_counts[mask])
    threshold = kwargs.get("threshold", 0.05)  # p‑value threshold
    return float(chi2), float(p) if p is not None else None, bool(p < threshold)


@METHOD_REGISTRY.register("cramers_v")
def cramers_v(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Cramér's V – association measure for categorical variables."""
    bins = 10
    try:
        ref_binned = np.digitize(
            ref, np.percentile(ref, np.linspace(0, 100, bins + 1)[1:-1])
        )
        cur_binned = np.digitize(
            cur, np.percentile(ref, np.linspace(0, 100, bins + 1)[1:-1])
        )
    except Exception:
        return 0.0, None, False
    all_cats = np.union1d(ref_binned, cur_binned)
    table = np.zeros((len(all_cats), 2))
    for i, cat in enumerate(all_cats):
        table[i, 0] = np.sum(ref_binned == cat)
        table[i, 1] = np.sum(cur_binned == cat)
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else 0
    threshold = kwargs.get("threshold", 0.1)
    return float(v), None, bool(v > threshold)


class DriftDetector:
    """
    Univariate + multivariate drift detector with second-order Hessian support.

    Usage:
        detector = DriftDetector(reference_data, config=DriftConfig(methods=["ks", "wasserstein_2"]))
        report = detector.detect(current_data)
    """

    def __init__(
        self, reference_data: Dict[str, np.ndarray], config: Any = None
    ) -> None:
        from .config import DriftConfig

        self.reference = {k: np.array(v) for k, v in reference_data.items()}
        self.config = config or DriftConfig()

    def detect(self, current_data: Dict[str, np.ndarray]) -> Any:
        """Detect drift across registered features and statistical methods."""
        from .report import DriftReport, FeatureDriftReport

        current = {k: np.array(v) for k, v in current_data.items()}
        features = self.config.features or list(self.reference.keys())
        all_feature_reports = []

        for feature in features:
            if feature not in self.reference or feature not in current:
                continue
            ref = self.reference[feature]
            cur = current[feature]

            if ref.dtype.kind not in "iuf" or cur.dtype.kind not in "iuf":
                continue

            method_reports = {}
            for method_key in self.config.methods:
                fn = METHOD_REGISTRY.get(method_key)
                if fn is None:
                    continue
                threshold = self.config.method_thresholds.get(
                    method_key, self.config.threshold
                )
                stat, p_val, drifted = fn(ref, cur, threshold=threshold)
                method_reports[method_key] = {
                    "statistic": float(stat),
                    "p_value": float(p_val) if p_val is not None else None,
                    "drifted": bool(drifted),
                    "threshold": float(threshold),
                }
            all_feature_reports.append(
                FeatureDriftReport(
                    feature=feature,
                    methods=method_reports,
                    reference_mean=float(ref.mean()),
                    current_mean=float(cur.mean()),
                    reference_std=float(ref.std()),
                    current_std=float(cur.std()),
                )
            )

        # Multivariate drift (optional)
        multivariate_result = None
        if self.config.enable_multivariate:
            multivariate_result = self._multivariate_drift(self.reference, current)

        # Concept drift
        concept_result = None
        if self.config.enable_concept_drift and "labels" in current:
            method_key = self.config.concept_metric
            fn = METHOD_REGISTRY.get(method_key)
            if fn:
                ref_lab = self.reference.get("labels", np.array([]))
                cur_lab = current["labels"]
                if ref_lab.size > 0 and cur_lab.size > 0:
                    stat, p, drifted = fn(
                        ref_lab, cur_lab, threshold=self.config.threshold
                    )
                    concept_result = {
                        "method": method_key,
                        "statistic": float(stat),
                        "drifted": bool(drifted),
                    }

        return DriftReport(
            features=all_feature_reports,
            multivariate=multivariate_result,
            concept_drift=concept_result,
            config=self.config,
        )

    def _multivariate_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """Multivariate drift detection using PCA, Domain Classifier, or Hessian Wasserstein-2."""
        multivariate_method = getattr(self.config, "multivariate_method", "pca")
        if multivariate_method == "domain_classifier":
            return self._domain_classifier_drift(ref, cur)
        if multivariate_method in (
            "hessian_wasserstein",
            "wasserstein_2",
            "wasserstein_2_nystrom",
            "fast_wasserstein",
        ):
            return self._hessian_wasserstein_drift(ref, cur)
        return self._pca_drift(ref, cur)

    def _hessian_wasserstein_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """Multivariate 2-Wasserstein Hessian / Covariance drift."""
        features = [
            f for f in (self.config.features or ref.keys()) if f in ref and f in cur
        ]
        if len(features) < 2:
            return {"method": "hessian_wasserstein", "error": "Not enough features"}
        X_ref = np.column_stack([ref[f] for f in features if ref[f].ndim == 1])
        X_cur = np.column_stack([cur[f] for f in features if cur[f].ndim == 1])

        multivariate_method = getattr(self.config, "multivariate_method", "hessian_wasserstein")
        if multivariate_method in ("wasserstein_2_nystrom", "fast_wasserstein"):
            stat, _, drifted = fast_wasserstein_detector(
                X_ref, X_cur, threshold=self.config.threshold
            )
            return {
                "method": "fast_wasserstein",
                "statistic": stat,
                "drifted": drifted,
                "threshold": self.config.threshold,
            }

        stat, _, drifted = hessian_wasserstein(
            X_ref, X_cur, threshold=self.config.threshold
        )
        return {
            "method": "hessian_wasserstein",
            "statistic": stat,
            "drifted": drifted,
            "threshold": self.config.threshold,
        }

    def _pca_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """PCA reconstruction error drift."""
        from sklearn.decomposition import PCA

        features = [
            f for f in (self.config.features or ref.keys()) if f in ref and f in cur
        ]
        if len(features) < 2:
            return {"method": "pca", "error": "Not enough features"}
        X_ref = np.column_stack([ref[f] for f in features if ref[f].ndim == 1])
        X_cur = np.column_stack([cur[f] for f in features if cur[f].ndim == 1])

        # Adaptive PCA component count to avoid n_components > min(n_samples, n_features)
        n_comp = min(
            self.config.multivariate_n_components, X_ref.shape[1], X_ref.shape[0]
        )
        pca = PCA(n_components=n_comp).fit(X_ref)
        err_ref = float(np.mean(pca.transform(X_ref) ** 2, axis=1).mean())
        err_cur = float(np.mean(pca.transform(X_cur) ** 2, axis=1).mean())
        drift_score = float(abs(err_cur - err_ref) / (err_ref + 1e-10))
        drifted = bool(drift_score > self.config.threshold)
        return {
            "method": "pca",
            "error_ref": err_ref,
            "error_cur": err_cur,
            "ratio": drift_score,
            "drifted": drifted,
        }

    def _domain_classifier_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """Domain classifier ROC-AUC drift."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        features = [
            f for f in (self.config.features or ref.keys()) if f in ref and f in cur
        ]
        if len(features) < 2:
            return {"method": "domain_classifier", "error": "Not enough features"}
        X_ref = np.column_stack([ref[f] for f in features if ref[f].ndim == 1])
        X_cur = np.column_stack([cur[f] for f in features if cur[f].ndim == 1])
        X = np.vstack([X_ref, X_cur])
        y = np.hstack([np.zeros(len(X_ref)), np.ones(len(X_cur))])

        auc = float(
            np.mean(
                cross_val_score(
                    LogisticRegression(max_iter=1000), X, y, cv=3, scoring="roc_auc"
                )
            )
        )
        drift_indicator = float(abs(auc - 0.5) * 2)
        drifted = bool(drift_indicator > self.config.threshold)
        return {
            "method": "domain_classifier",
            "auc": auc,
            "drift_score": drift_indicator,
            "drifted": drifted,
        }
