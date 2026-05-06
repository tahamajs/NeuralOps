from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse

def well_known_ignore(request):
    return HttpResponse(status=204)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('web_dashboard.urls')),
    path('api/', include('api.urls')),                  # ← this will pick up api/urls.py
    path('api/notes/', include('notes.urls')),
    path('api/model-versioning/', include('model_versioning.urls')),
    path('.well-known/appspecific/com.chrome.devtools.json', well_known_ignore),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)