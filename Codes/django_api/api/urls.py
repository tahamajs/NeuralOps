from django.urls import include, path
from . import views

urlpatterns = [
    # Authentication – must be first
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

    # System
    path("system/status/", views.system_status),
    path("system/gpu/", views.gpu_info),
    path("health/", views.health_check),
    path("version/", views.version),

    # Experiments (explicit)
    path("experiments/", views.experiment_list),
    path("experiments/<uuid:pk>/", views.experiment_detail),
    path("train/", views.start_train),
    path("evaluate/", views.start_evaluate),
    path("compress/", views.start_compress),
    path("runs/<uuid:pk>/", views.run_detail),
    path("runs/<uuid:pk>/stop/", views.stop_run),

    # Include remaining experiment‑specific patterns (no duplicates!)
    path('', include('experiment.urls')),
]