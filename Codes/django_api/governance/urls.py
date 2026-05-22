from django.urls import path
from . import views

urlpatterns = [
    # Audit
    path('audit/dashboard/', views.audit_log_dashboard, name='audit_dashboard'),
    # Compliance
    path('compliance/check/', views.trigger_compliance_check, name='trigger_compliance'),
    path('compliance/report/<uuid:report_id>/', views.compliance_report, name='compliance_report'),
    # Contracts
    path('contracts/', views.contract_list, name='contract_list'),
    path('contracts/<uuid:contract_id>/sign/', views.sign_contract, name='sign_contract'),
    # Drift
    path('drift/setup/<uuid:run_id>/', views.setup_drift_monitor, name='setup_drift'),
    path('drift/alerts/<uuid:monitor_id>/', views.drift_alerts, name='drift_alerts'),
    # Security
    path('security/scan/<uuid:version_id>/', views.scan_model, name='scan_model'),
    path('security/result/<uuid:scan_id>/', views.scan_result, name='scan_result'),
    # Observability
    path('monitor/setup/<uuid:version_id>/', views.setup_model_monitor, name='setup_monitor'),
    path('monitor/observations/<uuid:monitor_id>/', views.model_observations, name='observations'),
    # FGAC
    path('access/policies/', views.access_policy_list, name='access_policies'),
    path('access/grants/', views.grant_access, name='grant_access'),
]