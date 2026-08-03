"""FedAdam, FedYogi, FedAdagrad – Adaptive server optimizers (Reddi et al. 2020)."""
from typing import Any

import numpy as np

from .base import AGGREGATOR_REGISTRY, BaseAggregator


class FedOptAggregator(BaseAggregator):
    """Base class for FedOpt family (FedAdam, FedYogi, FedAdagrad)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.tau = 1e-3
        self.m = {}
        self.v = {}

    def aggregate(self, global_model, client_updates, client_weights, round_num):
        # Compute pseudo-gradient (difference from previous global)
        """Aggregate."""
        avg_update = self.compute_weighted_average(client_updates, client_weights)

        # Initialise momentum buffers if needed
        if not self.m:
            for k in avg_update:
                self.m[k] = np.zeros_like(avg_update[k])
                self.v[k] = np.zeros_like(avg_update[k])

        # Update momentum
        for k in avg_update:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * avg_update[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (avg_update[k] ** 2)

        # Apply adaptive rule and return global model
        new_global = {}
        for k in avg_update:
            new_global[k] = self.m[k] / (np.sqrt(self.v[k]) + self.tau)
        return new_global


@AGGREGATOR_REGISTRY.register("fedadam")
class FedAdamAggregator(FedOptAggregator):
    """FedAdam: β1=0.9, β2=0.99, τ=1e-3."""

    pass


@AGGREGATOR_REGISTRY.register("fedyogi")
class FedYogiAggregator(FedOptAggregator):
    """FedYogi: uses sign‑based adaptive update."""

    def aggregate(self, global_model, client_updates, client_weights, round_num) -> Any:
        """Aggregate."""
        avg_update = self.compute_weighted_average(client_updates, client_weights)
        if not self.m:
            for k in avg_update:
                self.m[k] = np.zeros_like(avg_update[k])
                self.v[k] = np.zeros_like(avg_update[k])

        for k in avg_update:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * avg_update[k]
            self.v[k] = self.v[k] - (1 - self.beta2) * (avg_update[k] ** 2) * np.sign(
                self.v[k] - avg_update[k] ** 2
            )

        new_global = {}
        for k in avg_update:
            new_global[k] = self.m[k] / (np.sqrt(self.v[k]) + self.tau)
        return new_global


@AGGREGATOR_REGISTRY.register("fedadagrad")
class FedAdagradAggregator(FedOptAggregator):
    """FedAdagrad: only second‑moment accumulator."""

    def aggregate(self, global_model, client_updates, client_weights, round_num) -> Any:
        """Aggregate."""
        avg_update = self.compute_weighted_average(client_updates, client_weights)
        if not self.v:
            for k in avg_update:
                self.v[k] = np.zeros_like(avg_update[k])

        for k in avg_update:
            self.v[k] += avg_update[k] ** 2

        new_global = {}
        for k in avg_update:
            new_global[k] = avg_update[k] / (np.sqrt(self.v[k]) + self.tau)
        return new_global
