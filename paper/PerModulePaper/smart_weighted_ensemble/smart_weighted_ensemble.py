r"""
Smart Weighted Ensemble (Alias for weighted_ensemble.py).
Re-exports SmartWeightedEnsemble, TaskAwareAttention, WeightedEnsembleAttention, and injection functions.
"""

from framework.attentions.weighted_ensemble import (
    SmartWeightedEnsemble,
    SmartWeightedEnsembleAttention,
    TaskAwareAttention,
    WeightedEnsembleAttention,
    inject_smart_weighted_ensemble,
    inject_weighted_ensemble,
    run_smart_ensemble,
)

__all__ = [
    "SmartWeightedEnsemble",
    "TaskAwareAttention",
    "WeightedEnsembleAttention",
    "SmartWeightedEnsembleAttention",
    "inject_weighted_ensemble",
    "inject_smart_weighted_ensemble",
    "run_smart_ensemble",
]
