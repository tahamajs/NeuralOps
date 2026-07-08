from abc import ABC, abstractmethod

from framework.core.registry import Registry

STRATEGY_REGISTRY = Registry("distributed_strategies")


class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def wrap_model(self, model, **kwargs):
        """Return the wrapped model."""
        pass

    def setup(self):
        """Set environment variables (MASTER_ADDR, WORLD_SIZE, etc.)."""
        pass

    def cleanup(self):
        """Destroy process group."""
        pass
