from abc import ABC, abstractmethod

from framework.core.registry import Registry

QUANTIZER_REGISTRY = Registry("quantizers")


class BaseQuantizer(ABC):
    @abstractmethod
    def apply(self, model, **kwargs):
        """Return the quantized model."""
        pass
