from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    @abstractmethod
    def train(self) -> dict[str, Any]:
        """Fit the model on the provided dataset."""
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Run evaluate and return results."""
        pass
