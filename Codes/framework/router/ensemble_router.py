"""
Ensemble router combining multiple routers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import defaultdict
from .base import BaseRouter, RouteResult


class EnsembleRouter(BaseRouter):
    """
    Ensemble router that aggregates predictions from multiple routers.
    
    Supported strategies:
    - weighted_vote: weighted sum of confidence scores
    - majority_vote: simple majority
    - stacking: train a meta-classifier (requires labeled data)
    - dynamic: router selection based on text characteristics
    """
    
    name = "ensemble_router"
    
    def __init__(
        self,
        routers: List[BaseRouter],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_vote",
        default_task: str = "factual",
        meta_classifier: Optional[Any] = None  # for stacking
    ):
        """
        Args:
            routers: List of router instances
            weights: Corresponding weights (must sum to 1 if given)
            strategy: 'weighted_vote', 'majority_vote', 'stacking', 'dynamic'
            default_task: Fallback task
            meta_classifier: For stacking, a classifier that takes router outputs as features
        """
        self.routers = routers
        self.strategy = strategy
        self.default_task = default_task
        self.meta_classifier = meta_classifier
        
        # Normalize weights
        if weights is None:
            weights = [1.0 / len(routers)] * len(routers)
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        # For dynamic strategy, need a selector router (could be a simple rule)
        self.selector: Optional[BaseRouter] = None
    
    def set_selector(self, selector: BaseRouter):
        """Set a router that decides which sub-router to use for each input (dynamic strategy)."""
        self.selector = selector
        self.strategy = "dynamic"
    
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        if self.strategy == "weighted_vote":
            return self._weighted_vote(text)
        elif self.strategy == "majority_vote":
            return self._majority_vote(text)
        elif self.strategy == "stacking":
            return self._stacking(text)
        elif self.strategy == "dynamic":
            return self._dynamic(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _weighted_vote(self, text: str) -> Tuple[str, Dict[str, float]]:
        all_scores = defaultdict(float)
        for router, weight in zip(self.routers, self.weights):
            task, scores = router.route(text)
            for t, score in scores.items():
                all_scores[t] += score * weight
        # Normalize to probabilities
        total = sum(all_scores.values())
        if total == 0:
            return self.default_task, {self.default_task: 1.0}
        probs = {k: v / total for k, v in all_scores.items()}
        best = max(probs, key=probs.get)
        return best, probs
    
    def _majority_vote(self, text: str) -> Tuple[str, Dict[str, float]]:
        votes = defaultdict(int)
        for router in self.routers:
            task, _ = router.route(text)
            votes[task] += 1
        best = max(votes, key=votes.get)
        # Confidence = fraction of votes
        confidence = votes[best] / len(self.routers)
        return best, {best: confidence}
    
    def _stacking(self, text: str) -> Tuple[str, Dict[str, float]]:
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier not set for stacking strategy.")
        # Build feature vector: for each router, get probability vector
        features = []
        task_set = set()
        for router in self.routers:
            task, scores = router.route(text)
            task_set.update(scores.keys())
        # Align tasks (use all tasks across all routers)
        tasks_sorted = sorted(list(task_set))
        for router in self.routers:
            _, scores = router.route(text)
            feat = [scores.get(t, 0.0) for t in tasks_sorted]
            features.extend(feat)
        # Meta-classifier predicts
        pred_idx = self.meta_classifier.predict([features])[0]
        pred_task = tasks_sorted[pred_idx] if pred_idx < len(tasks_sorted) else self.default_task
        # Get probabilities if meta-classifier supports it
        if hasattr(self.meta_classifier, 'predict_proba'):
            probs = self.meta_classifier.predict_proba([features])[0]
            scores = {t: probs[i] for i, t in enumerate(tasks_sorted)}
        else:
            scores = {pred_task: 1.0}
        return pred_task, scores
    
    def _dynamic(self, text: str) -> Tuple[str, Dict[str, float]]:
        if self.selector is None:
            raise ValueError("Selector router not set for dynamic strategy.")
        selected_idx, _ = self.selector.route(text)  # selects index as string
        try:
            idx = int(selected_idx)
        except ValueError:
            idx = 0
        idx = min(max(idx, 0), len(self.routers) - 1)
        return self.routers[idx].route(text)
    
    def can_route(self, text: str) -> bool:
        return any(r.can_route(text) for r in self.routers)
    
    def get_confidence(self, text: str) -> float:
        best, scores = self.route(text)
        return scores.get(best, 0.0)