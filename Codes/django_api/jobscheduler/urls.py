from django.urls import path
from . import views

urlpatterns = [
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<uuid:job_id>/pause/', views.job_pause, name='job_pause'),
    path('jobs/<uuid:job_id>/resume/', views.job_resume, name='job_resume'),
    path('jobs/<uuid:job_id>/stop/', views.job_stop, name='job_stop'),
    path('jobs/<uuid:job_id>/retry/', views.job_retry, name='job_retry'),
    path('jobs/<uuid:job_id>/history/', views.job_history, name='job_history'),
    path('jobs/<uuid:job_id>/notes/', views.job_notes, name='job_notes'),
    path('jobs/<uuid:job_id>/config/history/', views.get_config_history, name='config_history'),
    path('jobs/<uuid:job_id>/config/compare/<int:version_a>/<int:version_b>/', views.compare_configs, name='compare_configs'),
    path('jobs/<uuid:job_id>/metrics/', views.get_custom_metrics, name='get_custom_metrics'),
    path('jobs/<uuid:job_id>/metrics/log/', views.log_custom_metric, name='log_custom_metric'),
    path('jobs/<uuid:job_id>/environment/', views.get_environment, name='get_environment'),
    path('jobs/<uuid:job_id>/scheduling/', views.update_job_scheduling, name='update_job_scheduling'),
    path('jobs/stats/', views.job_stats, name='job_stats'),
    path('jobs/bulk-delete/', views.bulk_delete_jobs, name='bulk_delete_jobs'),

    path('scheduler/status/', views.scheduler_status, name='scheduler_status'),
    path('scheduler/metrics/', views.scheduler_metrics, name='scheduler_metrics'),
    path('system/gpu/', views.gpu_info),
    path('system/status/', views.system_status),
    path('my-jobs/', views.my_scheduled_jobs, name='my_scheduled_jobs'),

]