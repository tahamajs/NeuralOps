# framework/routing/__init__.py
from .base import BaseRouter, TrainableRouter, RouteResult
from .keyword_router import KeywordRouter
from .ml_router import TfidfRouter, FastTextRouter
from .ensemble_router import EnsembleRouter
from .router_registry import ROUTER_REGISTRY
from .router_manager import RouterManager

# The registry is now automatically populated because the classes
# are imported and decorated with @ROUTER_REGISTRY.register

__all__ = [
    'BaseRouter', 'TrainableRouter', 'RouteResult',
    'KeywordRouter', 'TfidfRouter', 'FastTextRouter', 'EnsembleRouter',
    'ROUTER_REGISTRY', 'RouterManager',
]