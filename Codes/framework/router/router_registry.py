"""
Router registry – centralized registration and lookup for all router implementations.

Maintains a global registry of available routers and provides factory methods.
"""

# framework/routing/router_registry.py
from typing import Dict, Optional, Type
from .base import BaseRouter


class RouterRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Type[BaseRouter]] = {}
        return cls._instance
    
    def register(self, name: Optional[str] = None):
        def decorator(cls):
            nonlocal name
            if name is None:
                name = getattr(cls, 'name', cls.__name__.lower())
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