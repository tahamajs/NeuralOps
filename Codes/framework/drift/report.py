"""Drift report data models."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeatureDriftReport:
    feature: str
    methods: Dict[
        str, Dict[str, Any]
    ]  # method_key → {statistic, p_value, drifted, threshold}
    reference_mean: float = 0.0
    current_mean: float = 0.0
    reference_std: float = 0.0
    current_std: float = 0.0


@dataclass
class DriftReport:
    features: List[FeatureDriftReport]
    multivariate: Optional[Dict[str, Any]] = None
    concept_drift: Optional[Dict[str, Any]] = None
    config: Any = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def n_features_drifted(self) -> int:
        return sum(
            1 for f in self.features if any(m["drifted"] for m in f.methods.values())
        )

    @property
    def n_features_tested(self) -> int:
        return len(self.features)

    @property
    def drift_percentage(self) -> float:
        n = self.n_features_tested
        return round(self.n_features_drifted / max(n, 1) * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "n_features_tested": self.n_features_tested,
            "n_features_drifted": self.n_features_drifted,
            "drift_percentage": self.drift_percentage,
            "features": [
                {
                    "feature": f.feature,
                    "reference_mean": f.reference_mean,
                    "current_mean": f.current_mean,
                    "methods": f.methods,
                }
                for f in self.features
            ],
            "multivariate": self.multivariate,
            "concept_drift": self.concept_drift,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "DriftReport":
        with open(path, "r") as f:
            data = json.load(f)
        features = [FeatureDriftReport(**f) for f in data.pop("features", [])]
        return cls(features=features, **data)
