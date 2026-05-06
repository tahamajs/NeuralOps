from django.urls import path
from . import views

urlpatterns = [
    path('runs/<uuid:run_id>/versions/', views.list_model_versions, name='list_model_versions'),
    path('versions/<uuid:version_id>/', views.model_version_detail, name='model_version_detail'),
    path('versions/<uuid:version_id>/publish/', views.publish_to_hub, name='publish_to_hub'),
    path('versions/<uuid:version_id>/download/', views.download_model_version, name='download_model_version'),
    path('versions/<uuid:version_id>/archive/', views.archive_to_cold_storage, name='archive_to_cold_storage'),
    path('versions/<uuid:version_id>/restore/', views.restore_from_cold_storage, name='restore_from_cold_storage'),
    path('runs/<uuid:run_id>/versions/cleanup/', views.cleanup_versions, name='cleanup_versions'),
    path('runs/<uuid:run_id>/versions/<int:version_number>/set_production/', views.set_production_version, name='set_production_version'),
    path('runs/<uuid:run_id>/versions/compare/<int:version_a>/<int:version_b>/', views.compare_versions, name='compare_versions'),
]