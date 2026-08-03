"""No distributed strategy – single GPU / CPU fallback."""
from typing import Any

from .base import STRATEGY_REGISTRY, BaseStrategy


@STRATEGY_REGISTRY.register("none")
class NoDistributed(BaseStrategy):
    def wrap_model(self, model, **kwargs) -> Any:
        """Wrap model."""
        return model

    def setup(self) -> None:
        """Initialise resources and configuration."""
        pass

    def cleanup(self) -> None:
        """Cleanup."""
        pass
