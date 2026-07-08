"""Base class for client selection strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List

from framework.core.registry import Registry

SELECTOR_REGISTRY = Registry("fl_client_selectors")


class BaseSelector(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def select(
        self, client_ids: List[int], round_history: List[Dict[str, float]]
    ) -> List[int]:
        """Return list of selected client IDs for the next round."""
        pass
