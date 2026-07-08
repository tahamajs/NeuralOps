# framework/routing/__init__.py
from .base import BaseRouter, RouteResult, TrainableRouter
from .ensemble_router import EnsembleRouter
from .keyword_router import KeywordRouter
from .ml_router import FastTextRouter, TfidfRouter
from .router_manager import RouterManager
from .router_registry import ROUTER_REGISTRY

# The registry is now automatically populated because the classes
# are imported and decorated with @ROUTER_REGISTRY.register

__all__ = [
    "BaseRouter",
    "TrainableRouter",
    "RouteResult",
    "KeywordRouter",
    "TfidfRouter",
    "FastTextRouter",
    "EnsembleRouter",
    "ROUTER_REGISTRY",
    "RouterManager",
]
