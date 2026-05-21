# portfolios/apps.py
from django.apps import AppConfig

class PortfoliosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'portfolios'      # ← fixed (was 'profiles')

    def ready(self):
        import portfolios.signals  # noqa