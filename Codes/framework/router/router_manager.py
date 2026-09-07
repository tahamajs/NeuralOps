"""
Router manager – lifecycle management and orchestration of routing systems.

Manages router creation, configuration, caching, and performance monitoring.
"""

# framework/routing/router_manager.py
import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import BaseRouter, RouteResult
from .keyword_router import KeywordRouter


class RouterManager:
    def __init__(
        self,
        default_router: Optional[BaseRouter] = None,
        cache_size: int = 10000,
        fallback_task: str = "factual",
    ) -> None:
        self.routers: Dict[str, BaseRouter] = {}
        self.active_router_name: Optional[str] = None
        self.fallback_task = fallback_task
        self.cache_size = cache_size
        self._cache: Dict[str, RouteResult] = {}
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time_ms": 0.0}
        )
        if default_router is None:
            default_router = KeywordRouter()
        self.register_router(default_router, make_active=True)

    def register_router(
        self, router: BaseRouter, name: Optional[str] = None, make_active: bool = False
    ) -> None:
        """Register router."""
        router_name = name or getattr(router, "name", router.__class__.__name__)
        self.routers[router_name] = router
        if make_active:
            self.active_router_name = router_name

    def route(
        self,
        text: str,
        router_name: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> RouteResult:
        """Route."""
        if router_name is None:
            router_name = self.active_router_name
        if router_name is None or router_name not in self.routers:
            raise ValueError(
                f"Router '{router_name}' not registered. Available: {list(self.routers.keys())}"
            )
        if use_cache and not force_refresh:
            cache_key = hashlib.md5(f"{text}:{router_name}".encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
        router = self.routers[router_name]
        start = time.perf_counter()
        result = router.route_with_metadata(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._metrics[router_name]["count"] += 1
        self._metrics[router_name]["total_time_ms"] += elapsed_ms
        if result.metadata is None:
            result.metadata = {}
        result.metadata["inference_time_ms"] = elapsed_ms
        if use_cache and result.confidence > 0.7:
            if len(self._cache) >= self.cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = result
        return result

    def batch_route(
        self,
        texts: List[str],
        router_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[RouteResult]:
        """Batch route."""
        return [self.route(t, router_name, use_cache) for t in texts]

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return the metrics."""
        metrics = {}
        for name, m in self._metrics.items():
            avg_time = m["total_time_ms"] / m["count"] if m["count"] > 0 else 0
            metrics[name] = {
                "calls": m["count"],
                "total_time_ms": m["total_time_ms"],
                "avg_time_ms": avg_time,
            }
        return metrics

    def clear_cache(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return the stats."""
        return {
            "active_router": self.active_router_name,
            "registered_routers": list(self.routers.keys()),
            "cache_size": len(self._cache),
            "metrics": self.get_metrics(),
        }

    def set_active_router(self, name: str) -> None:
        """Set the active router."""
        if name not in self.routers:
            raise ValueError(f"Router '{name}' not registered.")
        self.active_router_name = name

    def remove_router(self, name: str) -> None:
        """Remove router."""
        if name in self.routers:
            del self.routers[name]
            if self.active_router_name == name:
                self.active_router_name = (
                    next(iter(self.routers.keys())) if self.routers else None
                )

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics.clear()
