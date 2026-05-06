from abc import ABC, abstractmethod
from framework.core.registry import Registry

SERVER_REGISTRY = Registry("serving_backends")

class BaseServer(ABC):
    @abstractmethod
    def serve(self, model, **kwargs):
        """Start the inference server."""
        pass