"""Drift detector with a registry of statistical methods and stable 2-Wasserstein Hessian metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from framework.core.registry import Registry
from scipy import stats

METHOD_REGISTRY = Registry("drift_methods")


# ── Matrix Square Root & Wasserstein-2 Stability Helpers ────────────


def matrix_square_root(
    matrix: np.ndarray,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
) -> np.ndarray:
    """
    Computes the stable symmetric positive semi-definite matrix square root $A^{1/2}$.

    Applies Tikhonov damping:
        $$A_{\\text{reg}} = A + \\lambda I$$
    and clamps negative/sub-threshold eigenvalues to $1e-8$ to ensure strict positive semi-definiteness
    prior to square root calculation.

    Args:
        matrix: Input 2D square matrix (e.g. Hessian or Covariance).
        damping: Tikhonov regularization factor $\\lambda > 0$.
        min_eig: Minimum eigenvalue threshold for clamping (default 1e-8).

    Returns:
        np.ndarray: Matrix square root $A^{1/2}$.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    dim = matrix.shape[0]
    
    # FIX: Add Tikhonov damping (H = H + lambda * I) to prevent matrix ill-conditioning and singularity
    reg_matrix = matrix + damping * np.eye(dim, dtype=np.float64)
    
    # Symmetrize to prevent numerical asymmetry
    sym_matrix = 0.5 * (reg_matrix + reg_matrix.T)
    
    # Eigenvalue decomposition for symmetric/Hermitian matrices
    eigvals, eigvecs = np.linalg.eigh(sym_matrix)
    
    # FIX: Clamp negative and near-zero eigenvalues to 1e-8 before calculating matrix square roots
    eigvals_clamped = np.maximum(eigvals, min_eig)
    
    # Reconstruct matrix square root: A^{1/2} = V * diag(sqrt(lambda)) * V^T
    sqrt_matrix = eigvecs @ np.diag(np.sqrt(eigvals_clamped)) @ eigvecs.T
    return 0.5 * (sqrt_matrix + sqrt_matrix.T)


def bures_wasserstein_distance(
    cov1: np.ndarray,
    cov2: np.ndarray,
    mean1: Optional[np.ndarray] = None,
    mean2: Optional[np.ndarray] = None,
    damping: float = 1e-6,
    min_eig: float = 1e-8,
) -> float:
    """
    Computes the 2-Wasserstein (Bures-Wasserstein) metric between Gaussian distributions
    or second-order Hessian/covariance matrices:
        $$W_2^2(\\mu_1, \\Sigma_1; \\mu_2, \\Sigma_2) = \\|\\mu_1 - \\mu_2\\|_2^2 + \\text{Tr}\\left(\\Sigma_1 + \\Sigma_2 - 2 \\left(\\Sigma_1^{1/2} \\Sigma_2 \\Sigma_1^{1/2}\\right)^{1/2}\\right)$$

    Args:
        cov1: Reference covariance or Hessian matrix $\\Sigma_1 \\in \\mathbb{R}^{d \\times d}$.
        cov2: Current covariance or Hessian matrix $\\Sigma_2 \\in \\mathbb{R}^{d \\times d}$.
        mean1: Optional reference mean vector $\\mu_1 \\in \\mathbb{R}^d$.
        mean2: Optional current mean vector $\\mu_2 \\in \\mathbb{R}^d$.
        damping: Tikhonov regularization parameter $\\lambda$.
        min_eig: Eigenvalue clamp floor (1e-8).

    Returns:
        float: 2-Wasserstein distance.
    """
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

    # FIX: Compute square root of Sigma_1 with Tikhonov damping and eigenvalue clamping to 1e-8
    cov1_sqrt = matrix_square_root(cov1, damping=damping, min_eig=min_eig)
    
    # Inner product matrix M = Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2}
    inner_mat = cov1_sqrt @ cov2 @ cov1_sqrt
    
    # FIX: Compute square root of M with Tikhonov damping and eigenvalue clamping to 1e-8
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
        # FIX: Multivariate / Hessian 2-Wasserstein with Tikhonov damping and eigenvalue clamping to 1e-8
        stat = bures_wasserstein_distance(ref, cur, damping=damping, min_eig=min_eig)
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
        if getattr(self.config, "multivariate_method", "pca") == "domain_classifier":
            return self._domain_classifier_drift(ref, cur)
        if getattr(self.config, "multivariate_method", "pca") in ("hessian_wasserstein", "wasserstein_2"):
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
        
        stat, _, drifted = hessian_wasserstein(X_ref, X_cur, threshold=self.config.threshold)
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
        n_comp = min(self.config.multivariate_n_components, X_ref.shape[1], X_ref.shape[0])
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
