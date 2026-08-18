"""
Grassmann / Riemannian Manifold Federated Averaging for Orthogonal SVD Bases.

Aggregates orthogonal projection matrices and low-rank SVD components on the
Grassmann manifold Gr(d, k) using Riemannian logarithmic/exponential mappings and
Fréchet projection means rather than naive Euclidean linear averaging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import AGGREGATOR_REGISTRY, BaseAggregator

logger = logging.getLogger(__name__)


# ── Grassmannian Differential Geometry Primitives ───────────────────


def grassmann_log_map(base_u: np.ndarray, target_u: np.ndarray) -> np.ndarray:
    """
    Computes the Riemannian Logarithmic Map Log_{U_0}(U_1) on Grassmannian Gr(d, k).

    Args:
        base_u: Reference orthogonal basis U_0 in R^{d x k} with U_0^T U_0 = I_k.
        target_u: Target orthogonal basis U_1 in R^{d x k} with U_1^T U_1 = I_k.

    Returns:
        np.ndarray: Tangent vector Delta in T_{U_0}Gr(d, k) with Delta^T U_0 = 0.
    """
    # FIX: Compute principal angles and horizontal tangent projection on Grassmannian
    d, k = base_u.shape
    
    # Cross projection matrix
    m = base_u.T @ target_u  # (k, k)
    
    # SVD of cross matrix
    v0, cos_theta, v1_t = np.linalg.svd(m, full_matrices=False)
    
    # Principal angles: theta_j = arccos(s_j)
    cos_clamped = np.clip(cos_theta, -1.0, 1.0)
    thetas = np.arccos(cos_clamped)

    # Component orthogonal to base_u: (I - U_0 U_0^T) U_1
    orth_res = target_u - base_u @ m  # (d, k)
    
    # If target is already aligned with base
    if np.linalg.norm(orth_res) < 1e-11 or np.all(thetas < 1e-9):
        return np.zeros((d, k), dtype=base_u.dtype)

    v1 = v1_t.T
    orth_principal = orth_res @ v1  # (d, k)

    sin_theta = np.sin(thetas)
    # Compute scale factor theta / sin(theta) with limit 1 as theta -> 0
    scale = np.zeros_like(thetas)
    for j in range(k):
        if sin_theta[j] > 1e-7:
            scale[j] = thetas[j] / sin_theta[j]
        else:
            scale[j] = 1.0

    q_theta = orth_principal @ np.diag(scale)  # (d, k)
    delta = q_theta @ v0.T  # (d, k)
    return delta


def grassmann_exp_map(base_u: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Computes the Riemannian Exponential Map Exp_{U_0}(Delta) on Grassmannian Gr(d, k).

    Args:
        base_u: Base orthogonal basis U_0 in R^{d x k}.
        delta: Tangent vector in T_{U_0}Gr(d, k).

    Returns:
        np.ndarray: Transported orthogonal basis U_new in R^{d x k} on Gr(d, k).
    """
    # FIX: Use thin SVD of tangent vector to evaluate geodesic curve Exp_{U_0}(Delta)
    d, k = base_u.shape
    norm_delta = np.linalg.norm(delta)
    if norm_delta < 1e-12:
        return base_u.copy()

    u_delta, s_delta, v_delta_t = np.linalg.svd(delta, full_matrices=False)
    v_delta = v_delta_t.T
    cos_s = np.diag(np.cos(s_delta))
    sin_s = np.diag(np.sin(s_delta))

    # Geodesic on Grassmannian: U(t) = U_0 * V_Delta * cos(Sigma) + U_Delta * sin(Sigma)
    u_exp = base_u @ v_delta @ cos_s + u_delta @ sin_s
    
    # Orthonormalize via QR to eliminate numerical float drift
    q, r = np.linalg.qr(u_exp)
    # Fix sign flips from QR
    diag_sign = np.sign(np.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return q * diag_sign


def grassmann_karcher_mean(
    bases: List[np.ndarray],
    weights: np.ndarray,
    max_iter: int = 15,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Computes the Riemannian Karcher (Fréchet) mean on Gr(d, k) via iterative gradient descent.

    Args:
        bases: List of client orthogonal basis matrices U_i in R^{d x k}.
        weights: Normalized client weights summing to 1.
        max_iter: Maximum Riemannian optimization iterations.
        tol: Convergence tolerance for norm of averaged tangent vector.

    Returns:
        np.ndarray: Orthonormal Fréchet mean basis U_mean in R^{d x k}.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights /= np.sum(weights)

    # Initialize at highest-weighted client basis
    init_idx = int(np.argmax(weights))
    u_curr = bases[init_idx].copy()
    
    # Re-orthonormalize initial point
    q, _ = np.linalg.qr(u_curr)
    u_curr = q[:, : bases[0].shape[1]]

    for step in range(max_iter):
        avg_delta = np.zeros_like(u_curr)
        for i, u_i in enumerate(bases):
            # Log map from current iterate to client basis
            delta_i = grassmann_log_map(u_curr, u_i)
            avg_delta += weights[i] * delta_i

        delta_norm = float(np.linalg.norm(avg_delta))
        if delta_norm < tol:
            break

        # Step along geodesic in direction of average tangent vector
        u_curr = grassmann_exp_map(u_curr, avg_delta)

    return u_curr


def grassmann_frechet_mean_spectral(
    bases: List[np.ndarray],
    weights: np.ndarray,
    rank: Optional[int] = None,
) -> np.ndarray:
    """
    Computes the exact closed-form Projection (Chordal) Fréchet Mean on Gr(d, k).

    Under the chordal metric d_c^2(U_1, U_2) = 0.5 * ||U_1 U_1^T - U_2 U_2^T||_F^2,
    the optimal global minimizer is given by the top-k eigenvectors of the weighted
    projection matrix P_bar = sum(w_i * U_i U_i^T).

    Args:
        bases: List of client bases U_i in R^{d x k}.
        weights: Client weights.
        rank: Subspace rank k (inferred from bases[0] if None).

    Returns:
        np.ndarray: Orthonormal basis matrix U in R^{d x k}.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights /= np.sum(weights)
    d, k = bases[0].shape
    k = rank or k

    # FIX: Compute weighted average projection matrix P_bar = sum(w_i * U_i U_i^T)
    p_bar = np.zeros((d, d), dtype=np.float64)
    for i, u_i in enumerate(bases):
        # Ensure client basis is strictly orthonormal
        q, _ = np.linalg.qr(u_i)
        u_orth = q[:, :k]
        p_bar += weights[i] * (u_orth @ u_orth.T)

    # Symmetrize P_bar
    p_bar = 0.5 * (p_bar + p_bar.T)
    
    # Eigendecomposition: top-k eigenvectors form the Fréchet mean basis
    eigvals, eigvecs = np.linalg.eigh(p_bar)
    top_indices = np.argsort(eigvals)[-k:][::-1]
    u_mean = eigvecs[:, top_indices]
    return u_mean


def is_orthogonal_basis_tensor(param_name: str, tensor_arr: np.ndarray) -> bool:
    """
    Identifies whether a parameter tensor represents an orthogonal SVD basis.

    Args:
        param_name: Name of parameter in state dict.
        tensor_arr: Parameter numpy array.

    Returns:
        bool: True if tensor should be aggregated via Grassmann manifold mapping.
    """
    if tensor_arr.ndim != 2:
        return False
    
    d, k = tensor_arr.shape
    # If d < k, transpose to align with Gr(d, k)
    if d < k:
        tensor_arr = tensor_arr.T
        d, k = tensor_arr.shape

    if k < 2 or d < k:
        return False

    name_lower = param_name.lower()
    svd_keywords = [
        "u_proj",
        "v_proj",
        "svd_u",
        "svd_v",
        "u_basis",
        "v_basis",
        "basis_u",
        "basis_v",
        "orthogonal",
        "q_basis",
        "k_basis",
    ]
    if any(kw in name_lower for kw in svd_keywords):
        return True

    # Check numerical semi-orthogonality: ||U^T U - I_k||_F / k < 0.15
    gram = tensor_arr.T @ tensor_arr
    dev = np.linalg.norm(gram - np.eye(k)) / k
    return bool(dev < 0.15)


# ── Aggregator Strategy Implementation ──────────────────────────────


@AGGREGATOR_REGISTRY.register("grassmann_fedavg")
@AGGREGATOR_REGISTRY.register("riemannian_fedavg")
class GrassmannFedAvgAggregator(BaseAggregator):
    """
    Grassmann / Riemannian Federated Averaging Aggregator.

    Performs Riemannian / Projection Fréchet mean aggregation for orthogonal SVD bases
    while applying standard weighted Euclidean averaging to remaining parameters.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.method: str = getattr(config, "grassmann_method", "spectral")  # "spectral" or "karcher"

    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: np.ndarray,
        round_num: int,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Aggregates client update dictionaries into a global model update.

        Args:
            global_model: Global PyTorch model.
            client_updates: List of client delta dictionaries (client_state - global_state).
            client_weights: Array of client sample weights.
            round_num: Communication round index.

        Returns:
            Dict[str, np.ndarray]: Aggregated update delta dictionary.
        """
        if not client_updates:
            return {}

        total_weight = np.sum(client_weights)
        if total_weight == 0:
            total_weight = float(len(client_updates))
            client_weights = np.ones(len(client_updates), dtype=np.float64)
        norm_weights = client_weights / total_weight

        global_state = {k: v.detach().cpu().numpy() for k, v in global_model.state_dict().items()}
        aggregated_update = {}

        for key in client_updates[0]:
            is_basis = False
            if key in global_state:
                is_basis = is_orthogonal_basis_tensor(key, global_state[key])

            if is_basis:
                # FIX: Aggregate orthogonal SVD bases using Grassmann manifold Fréchet averaging
                d_orig, k_orig = global_state[key].shape
                transposed = False
                
                # Reconstruct full client weights: W_client = W_global + Delta_W
                client_full_bases = []
                for i in range(len(client_updates)):
                    full_base = global_state[key] + client_updates[i][key]
                    if d_orig < k_orig:
                        full_base = full_base.T
                        transposed = True
                    # Orthonormalize client basis before manifold aggregation
                    q, _ = np.linalg.qr(full_base)
                    client_full_bases.append(q[:, : min(d_orig, k_orig)])

                if self.method == "karcher":
                    agg_base = grassmann_karcher_mean(client_full_bases, norm_weights)
                else:
                    agg_base = grassmann_frechet_mean_spectral(client_full_bases, norm_weights)

                if transposed:
                    agg_base = agg_base.T

                # Global update delta on basis
                aggregated_update[key] = agg_base - global_state[key]
            else:
                # Standard weighted linear Euclidean average
                aggregated_update[key] = sum(
                    client_updates[i][key] * norm_weights[i]
                    for i in range(len(client_updates))
                )

        return aggregated_update
