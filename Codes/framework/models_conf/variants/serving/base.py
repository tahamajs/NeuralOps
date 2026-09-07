"""
Serving base – abstract interface for model serving implementations.

Defines the contract for serving implementations (Ray Serve, vLLM, TGI, Triton, etc.).
"""

from abc import ABC, abstractmethod

from framework.core.registry import Registry

SERVER_REGISTRY = Registry("serving_backends")


class BaseServer(ABC):
    @abstractmethod
    def serve(self, model, **kwargs) -> None:
        """Start the inference server."""
        pass
