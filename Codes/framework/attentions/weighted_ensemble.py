"""
Backward-compatible re-export module.

The original weighted_ensemble.py contained two unrelated 
classes: an ensemble classifier and a hybrid attention. They 
have been split into:

  - legacy_ensemble.py  (SmartWeightedEnsemble, TaskAwareAttention)
  - hybrid_attention.py (HybridAttention = WeightedEnsembleAttention)

This module preserves the old import paths.
"""

from framework.attentions.legacy_ensemble import (
    TaskAwareAttention,
    SmartWeightedEnsemble,
    run_smart_ensemble,
)
from framework.attentions.hybrid_attention import (
    HybridAttention,
    WeightedEnsembleAttention,
    SmartWeightedEnsembleAttention,
    inject_weighted_ensemble,
    inject_smart_weighted_ensemble,
)

__all__ = [
    "TaskAwareAttention",
    "SmartWeightedEnsemble",
    "run_smart_ensemble",
    "HybridAttention",
    "WeightedEnsembleAttention",
    "SmartWeightedEnsembleAttention",
    "inject_weighted_ensemble",
    "inject_smart_weighted_ensemble",
]
