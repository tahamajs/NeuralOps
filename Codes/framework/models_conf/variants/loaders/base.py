from abc import ABC, abstractmethod
from framework.core.registry import Registry

LOADER_REGISTRY = Registry("loaders")

class BaseLoader(ABC):
    @abstractmethod
    def load(self, model_name: str, num_labels: int = 2, **kwargs):
        """Return (model, tokenizer)."""
        pass