# experiment/apps.py
from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class ExperimentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'experiment'

    def ready(self):
        """
        Startup tasks: register signals, pre-load registries, etc.
        """
        # Import signals (if you have a signals.py file)
        try:
            from . import signals
        except ImportError:
            pass  # No signals module yet

        # Optional: warm up framework registries (lazy-loaded later)
        try:
            from framework.core.registry import ATTENTION_REGISTRY
            # Just touch it to load – not strictly necessary
            _ = len(ATTENTION_REGISTRY)
        except ImportError:
            logger.debug("Framework not available – some features disabled")

        logger.info("Experiment app ready")