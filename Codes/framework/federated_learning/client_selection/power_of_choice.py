"""Power‑of‑choice client selection – pick the best among a random subset."""

import random

from .base import SELECTOR_REGISTRY, BaseSelector


@SELECTOR_REGISTRY.register("power_of_choice")
class PowerOfChoiceSelector(BaseSelector):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.candidates = config.power_of_choice_candidates

    def select(self, client_ids, round_history) -> Any:
        """Select."""
        num = max(1, int(self.config.clients_per_round * self.config.num_clients))
        selected = []
        for _ in range(num):
            if not client_ids:
                break
            pool = random.sample(client_ids, min(self.candidates, len(client_ids)))
            # Pick the client with the best recent performance (if history available)
            if round_history:
                best = max(
                    pool, key=lambda cid: round_history[-1].get(f"client_{cid}_acc", 0)
                )
            else:
                best = random.choice(pool)
            selected.append(best)
            client_ids = [c for c in client_ids if c != best]
        return selected
