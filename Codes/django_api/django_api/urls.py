from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from paper import views as paper_views

def well_known_ignore(request):
    return HttpResponse(status=204)

from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('web_dashboard.urls')),         # Dashboard HTML pages
    path('api/', include('api.urls')),               # All API endpoints
    path('api/notes/', include('notes.urls')),
    path('api/model-versioning/', include('model_versioning.urls')),
    path('api/experiments/', include('experiment.urls')),
    path('api/jobs/', include('jobscheduler.urls')),
    path('api/agents/', include('agents.urls')),
    path('api/', include('sweeps.urls')),
    path('api/terminal/', include('terminal.urls')),  # Terminal REST API
    path('api/paper/', include('paper.urls')),         # Paper REST API
        path('api/profiles/', include('profiles.urls')),
    path('api/messaging/', include('messaging.urls')),
path('api/leaderboards/', include('leaderboards.urls')),

]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)