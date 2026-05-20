"""Configuration for drift detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DriftConfig:
    """Central configuration for drift detection."""

    # Methods to apply (keys from METHOD_REGISTRY)
    methods: List[str] = field(default_factory=lambda: ["ks", "psi", "wasserstein"])

    # Global threshold for flagging drift (per‑method thresholds override this)
    threshold: float = 0.1

    # Per‑method overrides
    method_thresholds: Dict[str, float] = field(default_factory=dict)

    # Multivariate drift
    enable_multivariate: bool = True
    multivariate_method: str = "pca"          # "pca" or "domain_classifier"
    multivariate_n_components: int = 5

    # Concept drift (prediction drift)
    enable_concept_drift: bool = False
    concept_metric: str = "psi"               # method used for prediction drift

    # Feature selection
    features: Optional[List[str]] = None      # if None, all numeric columns are used
    categorical_features: Optional[List[str]] = None
    max_categories: int = 20                  # above this, treat as numeric

    # Output
    generate_plots: bool = True
    output_dir: str = "./drift_reports"