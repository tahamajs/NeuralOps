"""
Celery application for the Django project.

Start the worker with:
    celery -A django_api worker -l info

Start the beat scheduler with:
    celery -A django_api beat -l info
"""

import os
from celery import Celery

# Set the default Django settings module for Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')

app = Celery('django_api')

# Load configuration from Django settings, prefixed with CELERY_
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks from all registered Django apps
app.autodiscover_tasks()


# ----------------------------------------------------------------------
# Optional: a simple debug task to verify Celery is working
# ----------------------------------------------------------------------
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')


# ----------------------------------------------------------------------
# Note: The periodic schedule is defined in settings.CELERY_BEAT_SCHEDULE.
# You can also override or extend it here via:
#     app.conf.beat_schedule = { ... }
# but the settings file is the recommended single source of truth.
# ----------------------------------------------------------------------