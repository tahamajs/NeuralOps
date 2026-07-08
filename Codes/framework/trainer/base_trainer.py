from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        pass
