from abc import ABC, abstractmethod
from framework.core.registry import Registry

PEFT_REGISTRY = Registry("peft_methods")

class BasePEFT(ABC):
    @abstractmethod
    def apply(self, model, **kwargs):
        """Return the adapted model."""
        pass