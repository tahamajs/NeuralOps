"""No distributed strategy – single GPU / CPU fallback."""

from .base import STRATEGY_REGISTRY, BaseStrategy


@STRATEGY_REGISTRY.register("none")
class NoDistributed(BaseStrategy):
    def wrap_model(self, model, **kwargs):
        return model

    def setup(self):
        pass

    def cleanup(self):
        pass
