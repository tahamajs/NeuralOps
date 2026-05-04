"""
Registry for routers.
"""

from typing import Dict, Optional, Type, Any
from .base import BaseRouter


class RouterRegistry:
    """Registry for router classes."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Type[BaseRouter]] = {}
        return cls._instance
    
    def register(self, name: Optional[str] = None):
        """Decorator to register a router class."""
        def decorator(cls):
            nonlocal name
            if name is None:
                name = cls.name if hasattr(cls, 'name') else cls.__name__.lower()
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Optional[Type[BaseRouter]]:
        return self._registry.get(name)
    
    def create(self, name: str, **kwargs) -> Optional[BaseRouter]:
        cls = self.get(name)
        if cls is None:
            return None
        return cls(**kwargs)
    
    def list(self) -> Dict[str, Type[BaseRouter]]:
        return self._registry.copy()


ROUTER_REGISTRY = RouterRegistry()