"""Base aggregator for FL strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, OrderedDict
import torch
import numpy as np
from framework.core.registry import Registry

AGGREGATOR_REGISTRY = Registry("fl_aggregators")

class BaseAggregator(ABC):
    """Interface for federated learning aggregation strategies."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def aggregate(
        self,
        global_model: torch.nn.Module,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: np.ndarray,
        round_num: int,
    ) -> OrderedDict:
        """
        Aggregate client updates into a new global model.

        Args:
            global_model: current global model
            client_updates: list of state dicts from selected clients
            client_weights: weight vector (e.g., dataset sizes) for each client
            round_num: current communication round

        Returns:
            New global model state dict
        """
        pass

    def compute_weighted_average(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Simple weighted average (FedAvg)."""
        total_weight = np.sum(client_weights)
        if total_weight == 0:
            total_weight = len(client_updates)
            client_weights = np.ones(len(client_updates))
        avg_update = {}
        for key in client_updates[0]:
            avg_update[key] = sum(
                client_updates[i][key] * client_weights[i] / total_weight
                for i in range(len(client_updates))
            )
        return avg_update