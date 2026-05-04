from django.apps import AppConfig

class JobschedulerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'jobscheduler'

    def ready(self):
        # Start the background scheduler thread when Django starts
        from .scheduler import scheduler
        # scheduler is a singleton; its thread starts automatically