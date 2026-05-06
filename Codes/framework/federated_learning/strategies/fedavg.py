"""FedAvg – the baseline federated averaging algorithm (McMahan et al. 2017)."""

from .base import BaseAggregator, AGGREGATOR_REGISTRY
import numpy as np

@AGGREGATOR_REGISTRY.register("fedavg")
class FedAvgAggregator(BaseAggregator):
    """Federated Averaging."""

    def aggregate(self, global_model, client_updates, client_weights, round_num):
        avg_update = self.compute_weighted_average(client_updates, client_weights)
        return avg_update