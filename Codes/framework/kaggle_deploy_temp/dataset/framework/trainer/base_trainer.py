from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Fit the model on the provided dataset."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluate and return results."""
        pass
