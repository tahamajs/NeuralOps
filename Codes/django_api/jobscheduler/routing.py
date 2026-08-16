from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/jobs/(?P<job_id>[^/]+)/log/$", consumers.JobLogConsumer.as_asgi()),
    re_path(r"ws/scheduler/jobs/(?P<job_id>[^/]+)/$", consumers.JobLogConsumer.as_asgi()),
]
