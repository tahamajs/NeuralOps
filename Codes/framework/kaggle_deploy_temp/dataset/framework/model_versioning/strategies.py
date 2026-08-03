import datetime
import logging
from typing import Any, List

from framework.core.registry import Registry

from .config import VersioningConfig

logger = logging.getLogger(__name__)
STRATEGY_REGISTRY = Registry("version_cleanup_strategies")


class BestKStrategy:
    """Keep the best K versions according to a primary metric (e.g., accuracy)."""

    def __init__(self, config: VersioningConfig) -> None:
        self.config = config

    @STRATEGY_REGISTRY.register("keep_best_k")
    def select_versions_to_delete(self, all_versions: List[Any]) -> List[Any]:
        """Select versions to delete."""
        if not all_versions:
            return []
        primary_metric = "eval_accuracy"  # could be configurable
        sorted_versions = sorted(
            all_versions,
            key=lambda v: v.metrics.get(primary_metric, 0),
            reverse=True,
        )
        return sorted_versions[self.config.keep_best_k :]


class TaggedStrategy:
    """Keep only versions that have specific tags (production, staging, etc.)."""

    def __init__(self, config: VersioningConfig) -> None:
        self.config = config

    @STRATEGY_REGISTRY.register("keep_tagged")
    def select_versions_to_delete(self, all_versions: List[Any]) -> List[Any]:
        """Select versions to delete."""
        protected = [
            v
            for v in all_versions
            if any(t in (v.tags or []) for t in self.config.keep_tags)
        ]
        return [v for v in all_versions if v not in protected]


class TimeBasedStrategy:
    """Keep versions created within the last N days."""

    def __init__(self, config: VersioningConfig) -> None:
        self.config = config

    @STRATEGY_REGISTRY.register("time_based")
    def select_versions_to_delete(self, all_versions: List[Any]) -> List[Any]:
        """Select versions to delete."""
        cutoff = datetime.datetime.now() - datetime.timedelta(
            days=self.config.time_days
        )
        return [v for v in all_versions if v.created_at < cutoff]


class SizeBasedStrategy:
    """Keep the smallest K versions by artifact size."""

    def __init__(self, config: VersioningConfig) -> None:
        self.config = config

    @STRATEGY_REGISTRY.register("size_based")
    def select_versions_to_delete(self, all_versions: List[Any]) -> List[Any]:
        """Select versions to delete."""
        if not all_versions:
            return []
        sorted_versions = sorted(all_versions, key=lambda v: v.artifact_size_mb)
        return sorted_versions[self.config.keep_best_k :]


class ManualStrategy:
    """Never delete automatically – user must explicitly remove versions."""

    def __init__(self, config: VersioningConfig) -> None:
        pass

    @STRATEGY_REGISTRY.register("manual")
    def select_versions_to_delete(self, all_versions: List[Any]) -> List[Any]:
        """Select versions to delete."""
        return []
