"""Differential Privacy – Gaussian noise addition (Abadi et al. 2016)."""

import numpy as np

from .base import PRIVACY_REGISTRY, BasePrivacyMechanism


@PRIVACY_REGISTRY.register("dp_sgd")
class DPSGDPrivacy(BasePrivacyMechanism):
    def apply(self, aggregated_update) -> Any:
        """Execute the apply operation and return results."""
        noise_scale = self.config.dp_noise_multiplier * self.config.dp_max_grad_norm
        dp_update = {}
        for k, v in aggregated_update.items():
            dp_update[k] = v + np.random.normal(0, noise_scale, v.shape)
        return dp_update
