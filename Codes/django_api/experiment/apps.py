import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ExperimentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "experiment"

    def ready(self):
        # Import signals if any
        try:
            from . import signals
        except ImportError:
            pass

        # Optional: warm up framework registries
        try:
            from framework.core.registry import ATTENTION_REGISTRY

            _ = len(ATTENTION_REGISTRY)
        except ImportError:
            logger.debug("Framework not available – some features disabled")

        logger.info("Experiment app ready")
