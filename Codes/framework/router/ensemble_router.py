"""
Ensemble router – combines multiple routers for improved routing decisions.

Aggregates predictions from multiple routing strategies.
"""

# framework/routing/ensemble_router.py
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseRouter
from .router_registry import ROUTER_REGISTRY


@ROUTER_REGISTRY.register("ensemble_router")
class EnsembleRouter(BaseRouter):
    name = "ensemble_router"

    def __init__(
        self,
        routers: List[BaseRouter],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_vote",
        default_task: str = "factual",
        meta_classifier: Optional[Any] = None,
    ) -> None:
        self.routers = routers
        self.strategy = strategy
        self.default_task = default_task
        self.meta_classifier = meta_classifier
        if weights is None:
            weights = [1.0 / len(routers)] * len(routers)
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.selector = None

    def set_selector(self, selector: BaseRouter) -> None:
        """Set the selector."""
        self.selector = selector
        self.strategy = "dynamic"

    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Route."""
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

    def _weighted_vote(self, text) -> Any:
        """ weighted vote."""
        all_scores = defaultdict(float)
        for router, w in zip(self.routers, self.weights):
            _, scores = router.route(text)
            for t, s in scores.items():
                all_scores[t] += s * w
        total = sum(all_scores.values())
        if total == 0:
            return self.default_task, {self.default_task: 1.0}
        probs = {k: v / total for k, v in all_scores.items()}
        best = max(probs, key=probs.get)
        return best, probs

    def _majority_vote(self, text) -> Any:
        """ majority vote."""
        votes = defaultdict(int)
        for router in self.routers:
            task, _ = router.route(text)
            votes[task] += 1
        best = max(votes, key=votes.get)
        confidence = votes[best] / len(self.routers)
        return best, {best: confidence}

    def _stacking(self, text) -> Any:
        """ stacking."""
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier not set for stacking.")
        # Build feature vector from all routers' probability outputs
        all_scores = []
        task_set = set()
        for router in self.routers:
            _, scores = router.route(text)
            task_set.update(scores.keys())
        tasks = sorted(task_set)
        for router in self.routers:
            _, scores = router.route(text)
            feat = [scores.get(t, 0.0) for t in tasks]
            all_scores.extend(feat)
        pred_idx = self.meta_classifier.predict([all_scores])[0]
        pred_task = tasks[pred_idx] if pred_idx < len(tasks) else self.default_task
        if hasattr(self.meta_classifier, "predict_proba"):
            probs = self.meta_classifier.predict_proba([all_scores])[0]
            scores = {t: probs[i] for i, t in enumerate(tasks)}
        else:
            scores = {pred_task: 1.0}
        return pred_task, scores

    def _dynamic(self, text) -> Any:
        """ dynamic."""
        if self.selector is None:
            raise ValueError("Selector router not set for dynamic strategy.")
        selected_idx, _ = self.selector.route(text)
        try:
            idx = int(selected_idx)
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(self.routers) - 1))
        return self.routers[idx].route(text)

    def can_route(self, text: str) -> bool:
        """Can route."""
        return any(r.can_route(text) for r in self.routers)
