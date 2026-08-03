from abc import ABC, abstractmethod

from framework.core.registry import Registry

STRATEGY_REGISTRY = Registry("distributed_strategies")


class BaseStrategy(ABC):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def wrap_model(self, model, **kwargs) -> None:
        """Return the wrapped model."""
        pass

    def setup(self) -> None:
        """Set environment variables (MASTER_ADDR, WORLD_SIZE, etc.)."""
        pass

    def cleanup(self) -> None:
        """Destroy process group."""
        pass
