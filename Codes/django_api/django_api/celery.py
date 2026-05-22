import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')

app = Celery('django_api')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


# Optional: define a debug task (useful for testing Celery is working)
@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')


# Optional: if you use Celery Beat, you can define the schedule here
# (or keep it in settings.CELERY_BEAT_SCHEDULE – your choice)
app.conf.beat_schedule = {
    'debug-every-10-seconds': {
        'task': 'django_api.celery.debug_task',
        'schedule': 10.0,
    },
}