# framework/routing/__init__.py
"""
Task routing system for dynamic model selection.

This module provides keyword-based, ML-based, and ensemble routing
to detect task types from input text and route to appropriate compressed models.
"""

from .keyword_router import KeywordRouter, detect_task_type, ROUTER_KEYWORDS
from .router_ensemble import RouterEnsemble, TaskRouter
from .ml_router import MLRouter, FastTextRouter
from .ensemble_router import EnsembleRouter

__all__ = [
    # Keyword router
    "KeywordRouter",
    "detect_task_type",
    "ROUTER_KEYWORDS",
    
    # Ensemble routers
    "RouterEnsemble",
    "TaskRouter",
    
    # ML routers
    "MLRouter",
    "FastTextRouter",
    
    # Combined
    "EnsembleRouter",
]

__version__ = "1.0.0"