"""
Routing module for task detection and text classification.
"""

from .base import BaseRouter, TrainableRouter, RouteResult
from .keyword_router import KeywordRouter, DEFAULT_KEYWORDS
from .ml_router import TfidfRouter, FastTextRouter
from .ensemble_router import EnsembleRouter
from .router_registry import ROUTER_REGISTRY
from .router_manager import RouterManager

__all__ = [
    'BaseRouter',
    'TrainableRouter',
    'RouteResult',
    'KeywordRouter',
    'DEFAULT_KEYWORDS',
    'TfidfRouter',
    'FastTextRouter',
    'EnsembleRouter',
    'ROUTER_REGISTRY',
    'RouterManager',
]