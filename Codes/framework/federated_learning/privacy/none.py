"""No privacy mechanism."""
from typing import Any

from .base import PRIVACY_REGISTRY, BasePrivacyMechanism


@PRIVACY_REGISTRY.register("none")
class NoPrivacy(BasePrivacyMechanism):
    def apply(self, aggregated_update) -> Any:
        """Execute the apply operation and return results."""
        return aggregated_update
