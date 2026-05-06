from django.urls import include, path
from . import views
from jobscheduler import views as job_views      # import the jobscheduler views

urlpatterns = [
    # Authentication
    path("auth/register/", views.register),
    path("auth/login/", views.login),
    path("auth/logout/", views.logout),
    path("auth/user/", views.user_info),
    path("auth/profile/", views.update_profile),
    path("auth/change-password/", views.change_password),
    path("auth/refresh-token/", views.refresh_token),

    # Configuration
    path("config/attentions/", views.list_attentions),
    path("config/compressors/", views.list_compressors),
    path("config/models/", views.list_models),
    path("config/tasks/", views.list_tasks),
    path("config/methods/", views.list_methods),
    path("config/presets/", views.list_presets),

    # Health / Version
    path("health/", views.health_check),
    path("version/", views.version),

    # ============ Jobscheduler – explicit paths (NO include) ============
    path('jobs/', job_views.job_list, name='job_list'),
    path('jobs/<uuid:job_id>/', job_views.job_detail, name='job_detail'),
    path('jobs/<uuid:job_id>/pause/', job_views.job_pause, name='job_pause'),
    path('jobs/<uuid:job_id>/resume/', job_views.job_resume, name='job_resume'),
    path('jobs/<uuid:job_id>/stop/', job_views.job_stop, name='job_stop'),
    path('jobs/<uuid:job_id>/retry/', job_views.job_retry, name='job_retry'),
    path('jobs/<uuid:job_id>/history/', job_views.job_history, name='job_history'),
    path('jobs/<uuid:job_id>/notes/', job_views.job_notes, name='job_notes'),
    path('jobs/<uuid:job_id>/config/history/', job_views.get_config_history, name='config_history'),
    path('jobs/<uuid:job_id>/config/compare/<int:version_a>/<int:version_b>/', job_views.compare_configs, name='compare_configs'),
    path('jobs/<uuid:job_id>/metrics/', job_views.get_custom_metrics, name='get_custom_metrics'),
    path('jobs/<uuid:job_id>/metrics/log/', job_views.log_custom_metric, name='log_custom_metric'),
    path('jobs/<uuid:job_id>/environment/', job_views.get_environment, name='get_environment'),

    path('scheduler/status/', job_views.scheduler_status, name='scheduler_status'),
    path('scheduler/metrics/', job_views.scheduler_metrics, name='scheduler_metrics'),
    path('system/gpu/', job_views.gpu_info),
    path('system/status/', job_views.system_status),

    # ============ Experiment leftovers (only runs/preview) ============
    path('', include('experiment.urls')),
]