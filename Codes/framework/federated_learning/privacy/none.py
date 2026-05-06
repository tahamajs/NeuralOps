"""No privacy mechanism."""

from .base import BasePrivacyMechanism, PRIVACY_REGISTRY

@PRIVACY_REGISTRY.register("none")
class NoPrivacy(BasePrivacyMechanism):
    def apply(self, aggregated_update):
        return aggregated_update