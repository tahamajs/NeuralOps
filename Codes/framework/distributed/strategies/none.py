"""No distributed strategy – single GPU / CPU."""

from .base import BaseStrategy, STRATEGY_REGISTRY

@STRATEGY_REGISTRY.register("none")
class NoDistributed(BaseStrategy):
    def wrap_model(self, model, **kwargs):
        return model