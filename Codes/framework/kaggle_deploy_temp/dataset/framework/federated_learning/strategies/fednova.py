"""FedNova – Normalized averaging with local steps (Wang et al. 2020)."""

import numpy as np

from .base import AGGREGATOR_REGISTRY, BaseAggregator


@AGGREGATOR_REGISTRY.register("fednova")
class FedNovaAggregator(BaseAggregator):
    """FedNova aggregation."""

    def aggregate(self, global_model, client_updates, client_weights, round_num):
        # Normalize by number of local steps (tau_i)
        # Here we assume client_weights includes tau_i
        """Aggregate."""
        normalized_updates = []
        normalized_weights = []
        for i, update in enumerate(client_updates):
            tau_i = client_weights[i]  # in FedNova, weight = tau_i (num local steps)
            norm_update = {k: v / tau_i for k, v in update.items()}
            normalized_updates.append(norm_update)
            normalized_weights.append(tau_i)
        normalized_weights = np.array(normalized_weights, dtype=float)
        return self.compute_weighted_average(normalized_updates, normalized_weights)
