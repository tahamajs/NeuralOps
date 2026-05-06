"""
Data Drift Detection – production‑grade, registry‑based system.

Supports:
  - Univariate drift: KS, PSI, Wasserstein, JS, Chi2, Cramér's V
  - Multivariate drift: PCA reconstruction error, domain classifier
  - Concept drift (prediction drift) via the same engine
  - Automatic integration with ResultsSaver and Django API
"""

from .config import DriftConfig
from .detectors import DriftDetector, METHOD_REGISTRY
from .report import DriftReport, FeatureDriftReport
from .visualization import generate_drift_plots
from .integration import auto_drift_check

__all__ = [
    "DriftConfig",
    "DriftDetector",
    "METHOD_REGISTRY",
    "DriftReport",
    "FeatureDriftReport",
    "generate_drift_plots",
    "auto_drift_check",
]