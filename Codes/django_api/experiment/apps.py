# experiments/apps.py
from django.apps import AppConfig

class ExperimentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'experiments'

    def ready(self):
        from .models import TrainingRun
        # Mark stale runs as stopped on startup
        TrainingRun.objects.filter(status='running').update(status='stopped')