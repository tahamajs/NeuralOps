"""
Base classes for the routing system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass, field


@dataclass
class RouteResult:
    """Result of a routing operation."""
    task: str
    confidence: float
    all_scores: Dict[str, float]
    router_name: str
    inference_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRouter(ABC):
    """Abstract base for all routers."""
    
    name: str = "base_router"
    
    @abstractmethod
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Route text to a task type.
        
        Returns:
            (predicted_task, confidence_scores_dict)
        """
        pass
    
    def route_with_metadata(self, text: str) -> RouteResult:
        """Route and return structured result with timing and metadata."""
        start = time.perf_counter()
        task, scores = self.route(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = scores.get(task, 0.0)
        # Normalize confidence if needed
        total = sum(scores.values())
        if total > 0:
            confidence = scores[task] / total
        return RouteResult(
            task=task,
            confidence=confidence,
            all_scores=scores,
            router_name=self.name,
            inference_time_ms=elapsed_ms
        )
    
    def batch_route(self, texts: List[str]) -> List[RouteResult]:
        """Route multiple texts efficiently."""
        return [self.route_with_metadata(t) for t in texts]
    
    @abstractmethod
    def can_route(self, text: str) -> bool:
        """Return True if this router can handle the text (e.g., length, language)."""
        return True


class TrainableRouter(BaseRouter, ABC):
    """Base for routers that require training."""
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[str], **kwargs) -> 'TrainableRouter':
        """Train the router on labeled data."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass