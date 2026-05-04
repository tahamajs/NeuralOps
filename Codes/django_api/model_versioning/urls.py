# model_versioning/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('runs/<uuid:run_id>/versions/', views.list_model_versions),
    path('versions/<uuid:version_id>/', views.model_version_detail),
    path('versions/<uuid:version_id>/publish/', views.publish_to_hub),
    path('versions/<uuid:version_id>/download/', views.download_model_version),
    path('versions/<uuid:version_id>/archive/', views.archive_to_cold_storage),
    path('runs/<uuid:run_id>/versions/cleanup/', views.cleanup_versions),
]