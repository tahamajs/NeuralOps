# framework/routing/ensemble_router.py
"""
Ensemble router combining multiple routing strategies.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .keyword_router import KeywordRouter
from .ml_router import MLRouter


class EnsembleRouter:
    """
    Ensemble router that combines multiple routing strategies.
    
    Uses weighted voting from keyword-based, ML-based, and other routers.
    
    Example:
        >>> keyword_router = KeywordRouter()
        >>> ml_router = MLRouter()
        >>> ml_router.train(train_texts, train_labels)
        >>> ensemble = EnsembleRouter([(keyword_router, 0.3), (ml_router, 0.7)])
        >>> task = ensemble.route("This is a test")
    """
    
    def __init__(
        self,
        routers: List[Tuple[any, float]],
        default_task: str = "factual"
    ):
        """
        Args:
            routers: List of (router_instance, weight) tuples
            default_task: Default task when no prediction available
        """
        self.routers = routers
        self.default_task = default_task
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize router weights to sum to 1."""
        total = sum(w for _, w in self.routers)
        if total > 0:
            self.routers = [(r, w / total) for r, w in self.routers]
    
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Route text using weighted ensemble of routers.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_task, confidence_scores)
        """
        all_scores = {}
        
        for router, weight in self.routers:
            if hasattr(router, 'route'):
                # Single prediction
                pred, scores = router.route(text)
                for task, score in scores.items():
                    all_scores[task] = all_scores.get(task, 0) + score * weight
            else:
                # Simple detect function
                pred = router.detect(text)
                all_scores[pred] = all_scores.get(pred, 0) + weight
        
        # Find best task
        best_task = max(all_scores, key=all_scores.get) if all_scores else self.default_task
        return best_task, all_scores
    
    def get_confidence(self, text: str) -> float:
        """Get confidence score for the best match."""
        best_task, scores = self.route(text)
        total = sum(scores.values())
        if total == 0:
            return 0.0
        return scores[best_task] / total