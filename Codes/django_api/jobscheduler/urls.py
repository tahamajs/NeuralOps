# jobscheduler/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Job CRUD and actions
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<uuid:job_id>/priority/', views.update_job_scheduling, name='job_priority'),
    path('jobs/<uuid:job_id>/scheduling/', views.update_job_scheduling, name='job_scheduling'),
    path('jobs/<uuid:job_id>/pause/', views.job_pause, name='job_pause'),
    path('jobs/<uuid:job_id>/resume/', views.job_resume, name='job_resume'),
    path('jobs/<uuid:job_id>/stop/', views.job_stop, name='job_stop'),
    path('jobs/<uuid:job_id>/retry/', views.job_retry, name='job_retry'),
    path('jobs/<uuid:job_id>/history/', views.job_history, name='job_history'),
    path('jobs/<uuid:job_id>/notes/', views.job_notes, name='job_notes'),

    # Custom metrics
    path('runs/<uuid:run_id>/metrics/', views.log_custom_metric, name='log_custom_metric'),
    path('runs/<uuid:run_id>/metrics/list/', views.get_custom_metrics, name='get_custom_metrics'),

    # Configuration versioning
    path('runs/<uuid:run_id>/config/history/', views.get_config_history, name='config_history'),
    path('runs/<uuid:run_id>/config/compare/<int:version_a>/<int:version_b>/', views.compare_configs, name='compare_configs'),

    # Environment & code versioning
    path('runs/<uuid:run_id>/environment/', views.get_environment, name='get_environment'),

    # System & scheduler
    path('system/gpu/', views.gpu_info, name='gpu_info'),
    path('system/status/', views.system_status, name='system_status'),
    path('scheduler/status/', views.scheduler_status, name='scheduler_status'),
    path('scheduler/metrics/', views.scheduler_metrics, name='scheduler_metrics'),

    # Optional: extra endpoints for future extensibility
    path('jobs/stats/', views.job_stats, name='job_stats'),          # summary statistics
    path('jobs/bulk/delete/', views.bulk_delete_jobs, name='bulk_delete_jobs'),  # admin utility
]