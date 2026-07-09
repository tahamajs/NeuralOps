import os
import django
import uuid
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")
django.setup()

from jobscheduler.models import Job

j = Job.objects.create()
j_id_str = str(j.id)
try:
    j_get = Job.objects.get(id=j_id_str)
    print("GET OK")
except Job.DoesNotExist:
    print("GET FAILED")
