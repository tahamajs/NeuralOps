"""Statistical drift detection engine using scipy and numpy."""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift, concept drift, and feature skew using statistical tests."""

    @staticmethod
    def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        Common in financial ML and monitoring.
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # Define bin edges using percentiles of expected
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges[-1] = np.inf  # make last bin include everything above

        # Count frequencies
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # Add small epsilon to avoid division by zero
        eps = 1e-10
        expected_pct = (expected_counts / max(len(expected), 1)) + eps
        actual_pct = (actual_counts / max(len(actual), 1)) + eps

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0, psi))

    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
        """Two-sample Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": p_value < 0.05,
        }

    @staticmethod
    def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
        """Wasserstein (Earth Mover's) distance between two distributions."""
        return float(stats.wasserstein_distance(reference, current))

    @staticmethod
    def js_divergence(reference: np.ndarray, current: np.ndarray) -> float:
        """Jensen-Shannon divergence."""
        # Convert to probability distributions
        eps = 1e-10
        ref_hist, bins = np.histogram(reference, bins=10, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        ref_hist = ref_hist + eps
        cur_hist = cur_hist + eps
        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()
        m = (ref_hist + cur_hist) / 2
        return float(stats.entropy(ref_hist, m) / 2 + stats.entropy(cur_hist, m) / 2)

    @staticmethod
    def chi2_test(reference: np.ndarray, current: np.ndarray) -> dict:
        """Chi-squared test for categorical features."""
        # Create contingency table from binned data
        bins = min(10, len(np.unique(reference)))
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        # Normalize to same total
        total = max(ref_hist.sum(), cur_hist.sum(), 1)
        ref_hist = ref_hist / ref_hist.sum() * total
        cur_hist = cur_hist / cur_hist.sum() * total
        # Remove zeros
        mask = (ref_hist > 0) & (cur_hist > 0)
        statistic, p_value = stats.chisquare(ref_hist[mask], cur_hist[mask])
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": p_value < 0.05,
        }

    def detect(self, metric: str, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Run the specified drift detection test.
        Returns the metric value (higher = more drift).
        """
        if metric == "psi":
            return self.compute_psi(reference, current)
        elif metric == "ks_test":
            result = self.ks_test(reference, current)
            return 1.0 - result["p_value"]  # convert to drift magnitude
        elif metric == "wasserstein":
            return self.wasserstein_distance(reference, current)
        elif metric == "js_divergence":
            return self.js_divergence(reference, current)
        elif metric == "chi2":
            result = self.chi2_test(reference, current)
            return 1.0 - result["p_value"]
        else:
            raise ValueError(f"Unknown metric: {metric}")
