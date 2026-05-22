from django.urls import path
from . import views

urlpatterns = [
    # Models
    path('models/', views.model_list, name='model_list'),
    path('models/<uuid:model_id>/', views.model_detail, name='model_detail'),
    path('models/<uuid:model_id>/versions/', views.version_list, name='version_list'),
    path('models/<uuid:model_id>/versions/upload/', views.upload_version, name='upload_version'),

    # Approval
    path('versions/<uuid:version_id>/request-approval/', views.request_approval, name='request_approval'),
    path('approvals/<uuid:request_id>/resolve/', views.resolve_approval, name='resolve_approval'),

    # Promotion
    path('versions/<uuid:version_id>/promote/', views.promote_version, name='promote_version'),

    # Access Control
    path('models/<uuid:model_id>/access/', views.access_policy_list, name='access_policy_list'),
    path('access/<uuid:policy_id>/revoke/', views.revoke_access, name='revoke_access'),

    # Audit
    path('models/<uuid:model_id>/audit/', views.audit_log, name='audit_log'),

    # Catalog & Download
    path('catalog/', views.catalog_search, name='catalog_search'),
    path('versions/<uuid:version_id>/download/', views.record_download, name='record_download'),
]