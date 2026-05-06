from django.urls import include, path
from . import views

urlpatterns = [
    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    path("auth/register/", views.register),
    path("auth/login/", views.login),
    path("auth/logout/", views.logout),
    path("auth/user/", views.user_info),
    path("auth/profile/", views.update_profile),
    path("auth/change-password/", views.change_password),
    path("auth/refresh-token/", views.refresh_token),

    # ------------------------------------------------------------------
    # Configuration – all available components
    # ------------------------------------------------------------------
    path("config/attentions/", views.list_attentions),
    path("config/compressors/", views.list_compressors),
    path("config/trainers/", views.list_trainers),
    path("config/methods/", views.list_methods),
    path("config/models/", views.list_models),
    path("config/tasks/", views.list_tasks),
    path("config/presets/", views.list_presets),
    path("config/losses/", views.list_losses),
    path("config/optimizers/", views.list_optimizers),
    path("config/schedulers/", views.list_schedulers),
    path("config/callbacks/", views.list_callbacks),
    path("config/augmentations/", views.list_augmentations),
    path("config/hpo/", views.hpo_options),
    path("config/distributed/", views.distributed_options),
    path("config/tracking/", views.tracking_options),
    path("config/deployment/", views.deployment_options),
    path("runs/<uuid:run_id>/save/", views.run_save),

    # ------------------------------------------------------------------
    # Advanced actions
    # ------------------------------------------------------------------
    path("hpo/start/", views.start_hpo_job),
    path("models/deploy/", views.deploy_model),
    path("security/sign/", views.sign_model),

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------
    path("health/", views.health_check),
    path("version/", views.version),

    # ------------------------------------------------------------------
    # Apps (jobs, experiments)
    # ------------------------------------------------------------------
    path('', include('jobscheduler.urls')),
    path('', include('experiment.urls')),
]