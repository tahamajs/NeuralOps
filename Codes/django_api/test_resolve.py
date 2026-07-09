import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")
django.setup()

from django.urls import resolve
try:
    match = resolve("/api/model-versioning/runs/12345678-1234-5678-1234-567812345678/versions/")
    print("RESOLVED VIEW:", match.func)
    print("RESOLVED ARGS:", match.args)
    print("RESOLVED KWARGS:", match.kwargs)
    print("RESOLVED ROUTE:", match.route)
except Exception as e:
    print("RESOLVE ERROR:", e)
