"""SCAFFOLD – Stochastic Controlled Averaging (Karimireddy et al. 2020)."""

import numpy as np

from .base import AGGREGATOR_REGISTRY, BaseAggregator


@AGGREGATOR_REGISTRY.register("scaffold")
class SCAFFOLDAggregator(BaseAggregator):
    """SCAFFOLD aggregation with control variates."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.server_control = {}
        self.client_controls = {}  # client_id -> control variate

    def aggregate(
        self, global_model, client_updates, client_weights, round_num, client_ids=None
    ) -> Any:
        """Aggregate."""
        avg_update = self.compute_weighted_average(client_updates, client_weights)

        # Update server control variate
        if not self.server_control:
            for k in avg_update:
                self.server_control[k] = np.zeros_like(avg_update[k])

        avg_client_control = {}
        if client_ids:
            selected_controls = [
                self.client_controls.get(cid, {}) for cid in client_ids
            ]
            for k in avg_update:
                avg_client_control[k] = np.mean(
                    [c.get(k, np.zeros_like(avg_update[k])) for c in selected_controls],
                    axis=0,
                )
        else:
            avg_client_control = {k: np.zeros_like(v) for k, v in avg_update.items()}

        # Server control update: c ← c + (1/N) * ∑(c_i - c)
        for k in avg_update:
            self.server_control[k] = (
                self.server_control[k] + avg_client_control[k] - avg_update[k]
            )

        return avg_update
