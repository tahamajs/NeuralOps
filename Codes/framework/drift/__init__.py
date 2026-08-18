"""
Data Drift Detection – production‑grade, registry‑based system.

Supports:
  - Univariate drift: KS, PSI, Wasserstein, JS, Chi2, Cramér's V
  - Multivariate drift: PCA reconstruction error, domain classifier
  - Concept drift (prediction drift) via the same engine
  - Automatic integration with ResultsSaver and Django API
"""

from .config import DriftConfig
from .detectors import (
    METHOD_REGISTRY,
    DriftDetector,
    bures_wasserstein_distance,
    fast_bures_wasserstein_nystrom,
    hutchinson_trace_estimator,
    matrix_square_root,
    nystrom_low_rank_approximation,
    nystrom_matrix_square_root,
)
from .integration import auto_drift_check
from .report import DriftReport, FeatureDriftReport
from .visualization import generate_drift_plots

__all__ = [
    "DriftConfig",
    "DriftDetector",
    "METHOD_REGISTRY",
    "DriftReport",
    "FeatureDriftReport",
    "generate_drift_plots",
    "auto_drift_check",
    "matrix_square_root",
    "bures_wasserstein_distance",
    "hutchinson_trace_estimator",
    "nystrom_low_rank_approximation",
    "nystrom_matrix_square_root",
    "fast_bures_wasserstein_nystrom",
]

