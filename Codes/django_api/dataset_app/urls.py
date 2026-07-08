from django.urls import path

from . import views

urlpatterns = [
    # Datasets
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/<uuid:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<uuid:dataset_id>/export/', views.dataset_export, name='dataset_export'),

    # Artifacts & Lineage
    path('artifacts/', views.artifact_list, name='artifact_list'),
    path('artifacts/<uuid:artifact_id>/lineage/', views.artifact_lineage, name='artifact_lineage'),

    # Audit Logs
    path('audit-logs/', views.audit_log_list, name='audit_log_list'),
    path('audit-logs/export/', views.audit_export_request, name='audit_export_request'),
]