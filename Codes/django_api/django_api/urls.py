from django.urls import path, include
from django.http import HttpResponse
from django.conf import settings
from pathlib import Path

def dashboard_view(request):
    """Serve the dashboard.html file directly."""
    dashboard_path = settings.STATICFILES_DIRS[0] / 'dashboard.html'
    if dashboard_path.exists():
        return HttpResponse(dashboard_path.read_text(), content_type='text/html')
    return HttpResponse("<h1>Dashboard not found</h1>", status=404)
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', include('web_dashboard.urls')),
    path('api/', include('api.urls')),
    path('.well-known/appspecific/com.chrome.devtools.json', lambda request: HttpResponse(status=204)),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)