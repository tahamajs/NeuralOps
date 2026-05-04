from django.apps import AppConfig

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        from .models import TrainingRun
        # Any run that was 'running' when the server stopped is now stale
        TrainingRun.objects.filter(status='running').update(status='stopped')