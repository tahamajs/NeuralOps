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






# Add these patterns inside your existing urlpatterns list

    # Load Balancer & GPU
    path("loadbalancer/gpu/status/", views.gpu_allocator_status),
    path("loadbalancer/gpu/nodes/", views.gpu_nodes_detail),
    path("loadbalancer/scheduler/status/", views.scheduler_status),
    path("loadbalancer/scheduler/queue/", views.scheduler_queue),
    path("loadbalancer/scheduler/history/", views.scheduler_history),
    path("loadbalancer/cluster/nodes/", views.cluster_nodes),
    path("loadbalancer/cluster/gpu-check/", views.cluster_gpu_availability),

    # Drift Detection
    path("runs/<uuid:run_id>/drift/", views.drift_report),
    path("runs/<uuid:run_id>/drift/check/", views.run_drift_check),

    # Fairness
    path("runs/<uuid:run_id>/fairness/", views.fairness_evaluation),

    # Robustness
    path("runs/<uuid:run_id>/robustness/", views.robustness_evaluation),

    # Prompts
    path("prompts/", views.prompt_list),
    path("prompts/<str:name>/", views.prompt_detail),
    path("prompts/<str:name>/history/", views.prompt_history),

    # HPO
    path("hpo/status/", views.hpo_status),

    # Federated Learning
    path("federated/status/", views.fl_status),
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