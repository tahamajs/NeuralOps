from django.apps import AppConfig

class JobschedulerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'jobscheduler'

    def ready(self):
        from .scheduler import scheduler
        # Start scheduler only once, e.g., in runserver or production worker
        # Avoid starting twice (e.g., in autoreload)
        import os
        if os.environ.get('RUN_MAIN') != 'true' and not scheduler.worker_thread:
            scheduler.start()