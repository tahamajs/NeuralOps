# jobscheduler/apps.py
from django.apps import AppConfig
import os

class JobschedulerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'jobscheduler'

    def ready(self):
        # Avoid starting scheduler in migrate or collectstatic
        if os.environ.get('RUN_MAIN') != 'true' and not os.environ.get('DJANGO_AUTORELOAD'):
            from .scheduler import scheduler
            # Ensure scheduler is started only once
            if not scheduler.worker_thread or not scheduler.worker_thread.is_alive():
                scheduler.start()