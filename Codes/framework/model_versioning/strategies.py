import datetime
import logging
from typing import List
from .config import VersioningConfig, CleanupPolicy

logger = logging.getLogger(__name__)

class BestKStrategy:
    """Keep the best K versions by a metric."""
    def __init__(self, config: VersioningConfig):
        self.config = config

    def select_versions_to_delete(self, all_versions):
        if not all_versions:
            return []
        sorted_versions = sorted(
            all_versions,
            key=lambda v: v.metrics.get('eval_accuracy', 0),
            reverse=True
        )
        return sorted_versions[self.config.keep_best_k:]


class TaggedStrategy:
    """Keep versions with specific tags (production, staging, etc.)."""
    def __init__(self, config: VersioningConfig):
        self.config = config

    def select_versions_to_delete(self, all_versions):
        protected = [v for v in all_versions if any(
            t in (v.tags or []) for t in self.config.keep_tags
        )]
        return [v for v in all_versions if v not in protected]


class TimeBasedStrategy:
    """Keep versions created within the last N days."""
    def __init__(self, config: VersioningConfig):
        self.config = config

    def select_versions_to_delete(self, all_versions):
        cutoff = datetime.datetime.now() - datetime.timedelta(days=self.config.keep_best_k)
        return [v for v in all_versions if v.created_at < cutoff]