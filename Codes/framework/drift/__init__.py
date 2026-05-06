# framework/drift/__init__.py
from .detector import DriftDetector, DriftReport, DriftMethod

# framework/drift/detector.py
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class DriftMethod(Enum):
    KS = "ks"                     # Kolmogorov-Smirnov
    PSI = "psi"                   # Population Stability Index
    WASSERSTEIN = "wasserstein"   # Earth Mover's Distance
    CHI2 = "chi2"                 # Chi-squared
    JS = "js"                     # Jensen-Shannon divergence

@dataclass
class DriftReport:
    feature: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float]
    drift_detected: bool
    threshold: float
    reference_dist: np.ndarray = field(repr=False)
    current_dist: np.ndarray = field(repr=False)

class DriftDetector:
    def __init__(self, reference_data: Dict[str, np.ndarray], threshold: float = 0.1):
        self.reference = {k: np.array(v) for k, v in reference_data.items()}
        self.threshold = threshold

    def detect(self, current_data: Dict[str, np.ndarray],
               methods: List[DriftMethod] = None) -> List[DriftReport]:
        if methods is None: methods = [DriftMethod.KS, DriftMethod.PSI]
        reports = []
        current = {k: np.array(v) for k, v in current_data.items()}
        for feature in self.reference:
            if feature not in current: continue
            ref = self.reference[feature]; cur = current[feature]
            for method in methods:
                stat, p_val, drifted = self._compute(ref, cur, method)
                reports.append(DriftReport(
                    feature=feature, method=method, statistic=stat,
                    p_value=p_val, drift_detected=drifted,
                    threshold=self.threshold, reference_dist=ref, current_dist=cur
                ))
        return reports

    def _compute(self, ref: np.ndarray, cur: np.ndarray, method: DriftMethod):
        if method == DriftMethod.KS:
            stat, p = stats.ks_2samp(ref, cur)
            return stat, p, stat > self.threshold
        elif method == DriftMethod.PSI:
            bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins='auto')
            ref_hist, _ = np.histogram(ref, bins=bins, density=True)
            cur_hist, _ = np.histogram(cur, bins=bins, density=True)
            ref_hist = np.clip(ref_hist, 1e-10, None); cur_hist = np.clip(cur_hist, 1e-10, None)
            psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
            return psi, None, psi > self.threshold
        elif method == DriftMethod.WASSERSTEIN:
            stat = stats.wasserstein_distance(ref, cur)
            return stat, None, stat > self.threshold
        elif method == DriftMethod.JS:
            from scipy.spatial.distance import jensenshannon
            bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins='auto')
            ref_hist, _ = np.histogram(ref, bins=bins, density=True)
            cur_hist, _ = np.histogram(cur, bins=bins, density=True)
            js = jensenshannon(ref_hist, cur_hist)
            return js, None, js > self.threshold
        return 0.0, None, False

    def summary(self, reports: List[DriftReport]) -> Dict[str, Any]:
        drifted = [r for r in reports if r.drift_detected]
        return {
            "total_features": len(set(r.feature for r in reports)),
            "drifted_features": len(set(r.feature for r in drifted)),
            "drift_percentage": round(len(drifted)/max(len(reports),1)*100, 1),
            "details": [{ "feature": r.feature, "method": r.method.value,
                          "stat": round(r.statistic,4), "drifted": r.drift_detected}
                        for r in reports],
        }