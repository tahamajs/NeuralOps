"""Drift detector with a registry of statistical methods."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from framework.core.registry import Registry
from scipy import stats

METHOD_REGISTRY = Registry("drift_methods")


# ── Method implementations ──────────────────────────────────────────


@METHOD_REGISTRY.register("ks")
def kolmogorov_smirnov(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Kolmogorov‑Smirnov test for continuous distributions."""
    stat, p = stats.ks_2samp(ref, cur)
    threshold = kwargs.get("threshold", 0.1)
    return stat, p, stat > threshold


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
    return psi, None, psi > threshold


@METHOD_REGISTRY.register("wasserstein")
def wasserstein(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Earth Mover's Distance (Wasserstein‑1)."""
    stat = stats.wasserstein_distance(ref, cur)
    threshold = kwargs.get("threshold", 0.1)
    return stat, None, stat > threshold


@METHOD_REGISTRY.register("js")
def jensen_shannon(
    ref: np.ndarray, cur: np.ndarray, **kwargs
) -> Tuple[float, Optional[float], bool]:
    """Jensen‑Shannon divergence."""
    bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins="auto")
    ref_hist, _ = np.histogram(ref, bins=bins, density=True)
    cur_hist, _ = np.histogram(cur, bins=bins, density=True)
    ref_hist = ref_hist / ref_hist.sum()
    cur_hist = cur_hist / cur_hist.sum()
    m = (ref_hist + cur_hist) / 2
    kl_ref = np.sum(ref_hist * np.log(ref_hist / (m + 1e-10) + 1e-10))
    kl_cur = np.sum(cur_hist * np.log(cur_hist / (m + 1e-10) + 1e-10))
    js = (kl_ref + kl_cur) / 2
    threshold = kwargs.get("threshold", 0.1)
    return js, None, js > threshold


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
    return chi2, p, p < threshold


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
    return v, None, v > threshold


class DriftDetector:
    """
    Univariate + multivariate drift detector.

    Usage:
        detector = DriftDetector(reference_data, method_keys=["ks","psi"])
        report = detector.detect(current_data)
    """

    def __init__(
        self, reference_data: Dict[str, np.ndarray], config: "DriftConfig" = None
    ) -> None:
        from .config import DriftConfig

        self.reference = {k: np.array(v) for k, v in reference_data.items()}
        self.config = config or DriftConfig()

    def detect(self, current_data: Dict[str, np.ndarray]) -> "DriftReport":
        """Detect."""
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
                    "drifted": drifted,
                    "threshold": threshold,
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
                        "statistic": stat,
                        "drifted": drifted,
                    }

        return DriftReport(
            features=all_feature_reports,
            multivariate=multivariate_result,
            concept_drift=concept_result,
            config=self.config,
        )

    def _multivariate_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """ multivariate drift."""
        if self.config.multivariate_method == "domain_classifier":
            return self._domain_classifier_drift(ref, cur)
        return self._pca_drift(ref, cur)

    def _pca_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """ pca drift."""
        from sklearn.decomposition import PCA

        features = [
            f for f in (self.config.features or ref.keys()) if f in ref and f in cur
        ]
        if len(features) < 2:
            return {"method": "pca", "error": "Not enough features"}
        X_ref = np.column_stack([ref[f] for f in features if ref[f].ndim == 1])
        X_cur = np.column_stack([cur[f] for f in features if cur[f].ndim == 1])
        pca = PCA(n_components=self.config.multivariate_n_components).fit(X_ref)
        err_ref = np.mean(pca.transform(X_ref) ** 2, axis=1).mean()
        err_cur = np.mean(pca.transform(X_cur) ** 2, axis=1).mean()
        drift_score = abs(err_cur - err_ref) / (err_ref + 1e-10)
        drifted = drift_score > self.config.threshold
        return {
            "method": "pca",
            "error_ref": err_ref,
            "error_cur": err_cur,
            "ratio": drift_score,
            "drifted": drifted,
        }

    def _domain_classifier_drift(self, ref: dict, cur: dict) -> Dict[str, Any]:
        """ domain classifier drift."""
        from sklearn.linear_model import LogisticRegression

        features = [
            f for f in (self.config.features or ref.keys()) if f in ref and f in cur
        ]
        if len(features) < 2:
            return {"method": "domain_classifier", "error": "Not enough features"}
        X_ref = np.column_stack([ref[f] for f in features])
        X_cur = np.column_stack([cur[f] for f in features])
        X = np.vstack([X_ref, X_cur])
        y = np.hstack([np.zeros(len(X_ref)), np.ones(len(X_cur))])
        from sklearn.model_selection import cross_val_score

        auc = np.mean(
            cross_val_score(
                LogisticRegression(max_iter=1000), X, y, cv=3, scoring="roc_auc"
            )
        )
        drift_indicator = abs(auc - 0.5) * 2
        drifted = drift_indicator > self.config.threshold
        return {
            "method": "domain_classifier",
            "auc": auc,
            "drift_score": drift_indicator,
            "drifted": drifted,
        }
