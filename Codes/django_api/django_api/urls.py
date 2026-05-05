# django_api/urls.py (main project)
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse

# ----------------------------------------------------------------------
# Custom .well-known route (silences browser DevTools probing)
# ----------------------------------------------------------------------
def well_known_ignore(request):
    return HttpResponse(status=204)

# ----------------------------------------------------------------------
# Main URL patterns
# ----------------------------------------------------------------------
urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),

    # Web dashboard (all HTML pages)
    path('', include('web_dashboard.urls')),

    # REST API endpoints (all under /api/)
    path('api/', include('experiment.urls')),           # ✅ singular, matches your app folder
    path('api/', include('jobscheduler.urls')),         # job scheduling, metrics
    path('api/', include('notes.urls')),                # researcher notebook
    path('api/', include('api.urls')),                  # authentication, system status
    path('api/', include('model_versioning.urls')),     # model version management (if exists)

    # Chrome DevTools probe (avoids console errors)
    path('.well-known/appspecific/com.chrome.devtools.json', well_known_ignore),
]

# ----------------------------------------------------------------------
# Serve static and media files in development (DEBUG=True)
# ----------------------------------------------------------------------
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)