# project/urls.py (main project – e.g., backend/urls.py)
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
    # Admin interface (optional, can be disabled in production)
    path('admin/', admin.site.urls),

    # Web dashboard (all HTML pages: /, /dashboard/, /login/, /register/, /statistics/, etc.)
    path('', include('web_dashboard.urls')),

    # REST API endpoints (all under /api/)
    path('api/', include('jobscheduler.urls')),      # job scheduling, metrics, config history
    path('api/', include('notes.urls')),             # researcher notebook (v1 endpoints)
    path('api/', include('api.urls')),               # authentication, system status, config lists
    path('api/', include('experiments.urls')),       # experiments, training runs, snapshots
    path('api/', include('model_versioning.urls')),  # model version management (add if exists)

    # Chrome DevTools probe (avoids console errors)
    path('.well-known/appspecific/com.chrome.devtools.json', well_known_ignore),
]

# ----------------------------------------------------------------------
# Serve static and media files in development (DEBUG=True)
# ----------------------------------------------------------------------
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# ----------------------------------------------------------------------
# Optional: custom error handlers (can be added if needed)
# ----------------------------------------------------------------------
# handler404 = 'web_dashboard.views.custom_404'
# handler500 = 'web_dashboard.views.custom_500'