"""Base class for privacy mechanisms."""

from abc import ABC, abstractmethod
from framework.core.registry import Registry

PRIVACY_REGISTRY = Registry("fl_privacy")

class BasePrivacyMechanism(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def apply(self, aggregated_update):
        """Apply privacy mechanism to the aggregated update."""
        pass