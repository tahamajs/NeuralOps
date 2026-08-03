"""MOON – Model-Contrastive Federated Learning (Li et al. 2021).

The contrastive loss is applied on the client side; the server aggregation is identical to FedAvg.
"""
from typing import Any

from .base import AGGREGATOR_REGISTRY, BaseAggregator


@AGGREGATOR_REGISTRY.register("moon")
class MOONAggregator(BaseAggregator):
    """MOON aggregation (identical to FedAvg on server side)."""

    def aggregate(self, global_model, client_updates, client_weights, round_num) -> Any:
        """Aggregate."""
        return self.compute_weighted_average(client_updates, client_weights)
