"""Quality‑based client selection – choose clients with highest recent accuracy."""
from typing import Any

import numpy as np

from .base import SELECTOR_REGISTRY, BaseSelector


@SELECTOR_REGISTRY.register("quality_based")
class QualityBasedSelector(BaseSelector):
    def select(self, client_ids, round_history) -> Any:
        """Select."""
        num = max(1, int(self.config.clients_per_round * self.config.num_clients))
        if not round_history:
            return list(
                np.random.choice(client_ids, min(num, len(client_ids)), replace=False)
            )
        # Gather last‑known accuracy per client
        scores = {}
        for cid in client_ids:
            key = f"client_{cid}_acc"
            scores[cid] = next(
                (r.get(key, 0) for r in reversed(round_history) if key in r), 0
            )
        sorted_clients = sorted(scores, key=scores.get, reverse=True)
        return sorted_clients[:num]
