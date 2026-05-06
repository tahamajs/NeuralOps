from django.urls import path
from . import views

urlpatterns = [
    # List versions for a run (with filtering, pagination, CSV export)
    path('runs/<uuid:run_id>/versions/', views.list_model_versions, name='list_model_versions'),
    
    # Single version detail
    path('versions/<uuid:version_id>/', views.model_version_detail, name='model_version_detail'),
    
    # Publish to Hugging Face Hub
    path('versions/<uuid:version_id>/publish/', views.publish_to_hub, name='publish_to_hub'),
    
    # Download model artifact
    path('versions/<uuid:version_id>/download/', views.download_model_version, name='download_model_version'),
    
    # Archive to cold storage
    path('versions/<uuid:version_id>/archive/', views.archive_to_cold_storage, name='archive_to_cold_storage'),
    
    # Restore from cold storage
    path('versions/<uuid:version_id>/restore/', views.restore_from_cold_storage, name='restore_from_cold_storage'),
    
    # Cleanup old versions for a run
    path('runs/<uuid:run_id>/versions/cleanup/', views.cleanup_versions, name='cleanup_versions'),
    
    # Set production version
    path('runs/<uuid:run_id>/versions/<int:version_number>/set_production/', views.set_production_version, name='set_production_version'),
    
    # Compare two versions
    path('runs/<uuid:run_id>/versions/compare/<int:version_a>/<int:version_b>/', views.compare_versions, name='compare_versions'),
]