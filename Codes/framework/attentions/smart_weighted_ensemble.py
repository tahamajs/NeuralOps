r"""
Smart Weighted Ensemble (Alias for weighted_ensemble.py).
Re-exports SmartWeightedEnsemble, TaskAwareAttention, and run_smart_ensemble.
"""

from framework.attentions.weighted_ensemble import (
    SmartWeightedEnsemble,
    TaskAwareAttention,
    run_smart_ensemble,
)

__all__ = ["SmartWeightedEnsemble", "TaskAwareAttention", "run_smart_ensemble"]
