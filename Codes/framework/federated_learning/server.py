"""FL Server orchestrator – runs rounds, selects clients, aggregates updates."""

import numpy as np
import torch
import logging
from typing import Dict, Any, List, Optional, Callable
from copy import deepcopy
from .config import FLConfig
from .strategies import AGGREGATOR_REGISTRY
from .client_selection import SELECTOR_REGISTRY
from .compression import COMPRESSION_REGISTRY
from .privacy import PRIVACY_REGISTRY

logger = logging.getLogger(__name__)


class FLServer:
    """
    Central FL server that coordinates training rounds.

    Usage:
        config = FLConfig(num_clients=10, num_rounds=50, aggregation="fedavg")
        server = FLServer(config, global_model)
        train_fn = lambda model, data: model  # user provides client training function
        server.run(client_data, train_fn)
    """

    def __init__(self, config: FLConfig, global_model: torch.nn.Module):
        self.config = config
        self.global_model = global_model
        self.aggregator = AGGREGATOR_REGISTRY.resolve(config.aggregation.value)(config)
        self.selector = SELECTOR_REGISTRY.resolve(config.client_selection.value)(config)
        self.compressor = COMPRESSION_REGISTRY.resolve(config.compression.value)(config)
        self.privacy = PRIVACY_REGISTRY.resolve(config.privacy.value)(config)

        self.round_history: List[Dict[str, float]] = []
        self.best_accuracy = 0.0

    def run(
        self,
        client_data: List[Any],
        train_fn: Callable[[torch.nn.Module, Any], torch.nn.Module],
        eval_fn: Optional[Callable[[torch.nn.Module], Dict[str, float]]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run federated training.

        Args:
            client_data: list of data objects (one per client)
            train_fn: function (model, data) -> updated model (state dict or model)
            eval_fn: optional evaluation function (model) -> {"accuracy": ..., "loss": ...}
            progress_callback: optional callback for dashboard updates

        Returns:
            dict with final metrics and round history
        """
        for round_num in range(1, self.config.num_rounds + 1):
            # 1. Select clients
            selected_indices = self.selector.select(
                list(range(self.config.num_clients)),
                self.round_history,
            )
            num_selected = max(1, int(self.config.clients_per_round * self.config.num_clients))
            selected_indices = selected_indices[:num_selected]

            # 2. Send model to clients, train locally, collect updates
            client_updates = []
            client_weights = []
            for idx in selected_indices:
                # Clone global model for this client
                client_model = deepcopy(self.global_model)
                # Train locally
                updated_model = train_fn(client_model, client_data[idx])
                # Compute update (difference from global)
                update = self._compute_update(self.global_model, updated_model)
                # Compress
                compressed = self.compressor.compress(update)
                # Decompress on server side (for aggregation)
                decompressed = self.compressor.decompress(compressed)
                client_updates.append(decompressed)
                client_weights.append(self._get_client_weight(idx, client_data[idx]))

            # 3. Aggregate (server‑side computation)
            new_state = self.aggregator.aggregate(
                self.global_model, client_updates,
                np.array(client_weights), round_num,
                client_ids=selected_indices,
            )

            # 4. Privacy mechanism
            new_state = self.privacy.apply(new_state)

            # 5. Update global model
            self._apply_update(self.global_model, new_state)

            # 6. Evaluate
            round_metrics = {}
            if eval_fn and round_num % self.config.report_every_rounds == 0:
                round_metrics = eval_fn(self.global_model)
                logger.info("Round %d: %s", round_num, round_metrics)
            self.round_history.append(round_metrics)

            # 7. Progress callback
            if progress_callback:
                progress_callback(round_num, self.config.num_rounds, round_metrics)

        # Final evaluation
        final_metrics = eval_fn(self.global_model) if eval_fn else {}
        final_metrics["round_history"] = self.round_history
        return final_metrics

    def _compute_update(self, global_model, client_model):
        """Compute the difference between client and global model."""
        global_state = global_model.state_dict()
        client_state = client_model.state_dict() if isinstance(client_model, torch.nn.Module) else client_model
        update = {}
        for k in global_state:
            if k in client_state:
                update[k] = (client_state[k].cpu().numpy() - global_state[k].cpu().numpy())
        return update

    def _apply_update(self, global_model, update):
        """Apply aggregated update to the global model."""
        state = global_model.state_dict()
        for k in update:
            if k in state:
                state[k] += torch.tensor(update[k], dtype=state[k].dtype)
        global_model.load_state_dict(state)

    def _get_client_weight(self, idx, data):
        """Return client weight (default: dataset size)."""
        if hasattr(data, '__len__'):
            return len(data)
        return 1