from django.urls import path
from . import views

urlpatterns = [
    path('api/jobs/', views.job_list, name='job_list'),
    path('api/jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('api/jobs/<uuid:job_id>/pause/', views.job_pause, name='job_pause'),
    path('api/jobs/<uuid:job_id>/resume/', views.job_resume, name='job_resume'),
    path('api/jobs/<uuid:job_id>/stop/', views.job_stop, name='job_stop'),
    path('api/jobs/<uuid:job_id>/retry/', views.job_retry, name='job_retry'),
    path('api/jobs/<uuid:job_id>/history/', views.job_history, name='job_history'),
    path('api/jobs/<uuid:job_id>/notes/', views.job_notes, name='job_notes'),
    path('api/jobs/<uuid:job_id>/config/history/', views.get_config_history, name='config_history'),
    path('api/jobs/<uuid:job_id>/config/compare/<int:version_a>/<int:version_b>/', views.compare_configs, name='compare_configs'),
    path('api/jobs/<uuid:job_id>/metrics/', views.get_custom_metrics, name='get_custom_metrics'),
    path('api/jobs/<uuid:job_id>/metrics/log/', views.log_custom_metric, name='log_custom_metric'),
    path('api/jobs/<uuid:job_id>/environment/', views.get_environment, name='get_environment'),
    path('api/jobs/<uuid:job_id>/scheduling/', views.update_job_scheduling, name='update_job_scheduling'),
    path('api/jobs/stats/', views.job_stats, name='job_stats'),
    path('api/jobs/bulk-delete/', views.bulk_delete_jobs, name='bulk_delete_jobs'),
    path('api/scheduler/status/', views.scheduler_status, name='scheduler_status'),
    path('api/scheduler/metrics/', views.scheduler_metrics, name='scheduler_metrics'),
    path('api/system/gpu/', views.gpu_info, name='gpu_info'),
    path('api/system/status/', views.system_status, name='system_status'),
]