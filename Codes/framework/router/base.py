# framework/routing/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
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
    """Abstract base class for all routers."""
    
    name: str = "base_router"
    
    @abstractmethod
    def route(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Route text to a task type."""
        pass
    
    def route_with_metadata(self, text: str) -> RouteResult:
        start = time.perf_counter()
        task, scores = self.route(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = scores.get(task, 0.0)
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
        return [self.route_with_metadata(t) for t in texts]
    
    @abstractmethod
    def can_route(self, text: str) -> bool:
        """Return True if this router can handle the text."""
        return True


class TrainableRouter(BaseRouter, ABC):
    """Base for routers that require training."""
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[str], **kwargs) -> 'TrainableRouter':
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass