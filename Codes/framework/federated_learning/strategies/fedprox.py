"""FedProx – Proximal term to handle heterogeneity (Li et al. 2020)."""

from .base import AGGREGATOR_REGISTRY, BaseAggregator


@AGGREGATOR_REGISTRY.register("fedprox")
class FedProxAggregator(BaseAggregator):
    """Federated Proximal (FedProx)."""

    def aggregate(self, global_model, client_updates, client_weights, round_num):
        # FedProx uses same server aggregation as FedAvg; the prox term is applied on client side
        return self.compute_weighted_average(client_updates, client_weights)
