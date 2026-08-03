import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")
django.setup()

from experiment.models import TrainingRun
from jobscheduler.models import Job

print("Job objects count:", Job.objects.count())
print("TrainingRun objects count:", TrainingRun.objects.count())
