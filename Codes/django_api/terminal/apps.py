from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class TerminalConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'terminal'

    def ready(self):
        # Import signals only if the module exists
        try:
            import terminal.signals  # noqa
        except ImportError:
            logger.debug("terminal.signals not found – signals disabled")