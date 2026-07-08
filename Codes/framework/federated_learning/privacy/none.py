"""No privacy mechanism."""

from .base import PRIVACY_REGISTRY, BasePrivacyMechanism


@PRIVACY_REGISTRY.register("none")
class NoPrivacy(BasePrivacyMechanism):
    def apply(self, aggregated_update):
        return aggregated_update
