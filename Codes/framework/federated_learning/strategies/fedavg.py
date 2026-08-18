"""FedAvg – the baseline federated averaging algorithm (McMahan et al. 2017)."""
from typing import Any, Dict, List

import numpy as np
import torch

from .base import AGGREGATOR_REGISTRY, BaseAggregator


@AGGREGATOR_REGISTRY.register("fedavg")
class FedAvgAggregator(BaseAggregator):
    """Federated Averaging baseline aggregator."""

    def aggregate(
        self,
        global_model: torch.nn.Module,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: np.ndarray,
        round_num: int,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using standard weighted arithmetic averaging."""
        avg_update = self.compute_weighted_average(client_updates, client_weights)
        return avg_update
