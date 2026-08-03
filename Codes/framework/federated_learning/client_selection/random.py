"""Random client selection."""

import random

from .base import SELECTOR_REGISTRY, BaseSelector


@SELECTOR_REGISTRY.register("random")
class RandomSelector(BaseSelector):
    def select(self, client_ids, round_history) -> Any:
        """Select."""
        num = max(1, int(self.config.clients_per_round * self.config.num_clients))
        return random.sample(client_ids, min(num, len(client_ids)))
