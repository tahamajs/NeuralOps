from api import views, views_attention_dashboard
from django.urls import path

urlpatterns = [
    # Auth
    path("auth/register/", views.register, name="register"),
    path("auth/login/", views.login, name="login"),
    path("auth/logout/", views.logout, name="logout"),
    path("auth/user/", views.user_info, name="user_info"),
    path("auth/profile/", views.update_profile, name="update_profile"),
    path("auth/change-password/", views.change_password, name="change_password"),
    path("auth/refresh-token/", views.refresh_token, name="refresh_token"),
    # Configuration
    path("config/attentions/", views.list_attentions, name="list_attentions"),
    path("config/compressors/", views.list_compressors, name="list_compressors"),
    path("config/trainers/", views.list_trainers, name="list_trainers"),
    path("config/methods/", views.list_methods, name="list_methods"),
    path("config/models/", views.list_models, name="list_models"),
    path("config/tasks/", views.list_tasks, name="list_tasks"),
    path("config/presets/", views.list_presets, name="list_presets"),
    path("config/losses/", views.list_losses, name="list_losses"),
    path("config/optimizers/", views.list_optimizers, name="list_optimizers"),
    path("config/schedulers/", views.list_schedulers, name="list_schedulers"),
    path("config/callbacks/", views.list_callbacks, name="list_callbacks"),
    path("config/augmentations/", views.list_augmentations, name="list_augmentations"),
    path("config/variants/", views.list_model_variants, name="list_model_variants"),
    path("config/hpo/", views.hpo_options, name="hpo_options"),
    path("config/distributed/", views.distributed_options, name="distributed_options"),
    path("config/tracking/", views.tracking_options, name="tracking_options"),
    path("config/deployment/", views.deployment_options, name="deployment_options"),
    path(
        "config/search-models/",
        views.search_huggingface_models,
        name="search_huggingface_models",
    ),
    # Mega Launcher
    path("launch/", views.launch_experiment, name="launch_experiment"),
    # Load Balancer / GPU
    path(
        "loadbalancer/gpu/status/",
        views.gpu_allocator_status,
        name="gpu_allocator_status",
    ),
    path("loadbalancer/gpu/nodes/", views.gpu_nodes_detail, name="gpu_nodes_detail"),
    path(
        "loadbalancer/scheduler/status/",
        views.scheduler_status,
        name="scheduler_status",
    ),
    path(
        "loadbalancer/scheduler/queue/", views.scheduler_queue, name="scheduler_queue"
    ),
    path(
        "loadbalancer/scheduler/history/",
        views.scheduler_history,
        name="scheduler_history",
    ),
    path("loadbalancer/cluster/nodes/", views.cluster_nodes, name="cluster_nodes"),
    path(
        "loadbalancer/cluster/gpu-check/",
        views.cluster_gpu_availability,
        name="cluster_gpu_availability",
    ),
    # Drift / Fairness / Robustness
    path("runs/<uuid:run_id>/drift/", views.drift_report, name="drift_report"),
    path(
        "runs/<uuid:run_id>/drift/check/", views.run_drift_check, name="run_drift_check"
    ),
    path(
        "runs/<uuid:run_id>/fairness/",
        views.fairness_evaluation,
        name="fairness_evaluation",
    ),
    path(
        "runs/<uuid:run_id>/robustness/",
        views.robustness_evaluation,
        name="robustness_evaluation",
    ),
    # Prompts
    path("prompts/", views.prompt_list, name="prompt_list"),
    path("prompts/<str:name>/", views.prompt_detail, name="prompt_detail"),
    path("prompts/<str:name>/history/", views.prompt_history, name="prompt_history"),
    # HPO / FL status
    path("hpo/status/", views.hpo_status, name="hpo_status"),
    path("federated/status/", views.fl_status, name="fl_status"),
    # Health & System
    path("health/", views.health_check, name="health_check"),
    path("version/", views.version, name="version"),
    # Model Cards
    path(
        "runs/<uuid:run_id>/model-card/",
        views.generate_model_card_view,
        name="model_card",
    ),
    path(
        "runs/<uuid:run_id>/model-card/push/",
        views.push_model_card_view,
        name="push_model_card",
    ),
    # Lineage
    path("runs/<uuid:run_id>/lineage/", views.data_lineage_view, name="data_lineage"),
    path(
        "models/<uuid:version_id>/lineage/",
        views.model_lineage_view,
        name="model_lineage_graph",
    ),
    # Global Search
    path("search/", views.global_search, name="global_search"),
    # Activity Feed
    path("activity/", views.activity_feed, name="activity_feed"),
    # Experiments CRUD (delegated to experiment app via this API)
    path("experiments/", views.experiment_list, name="experiment_list"),
    path("experiments/<uuid:pk>/", views.experiment_detail, name="experiment_detail"),
    path("experiments/start_train/", views.start_train, name="start_train"),
    path("experiments/start_evaluate/", views.start_evaluate, name="start_evaluate"),
    path("experiments/start_compress/", views.start_compress, name="start_compress"),
    path("train/", views.start_train, name="train"),
    path("evaluate/", views.start_evaluate, name="evaluate"),
    path("compress/", views.start_compress, name="compress"),
    path("runs/<uuid:pk>/", views.run_detail, name="run_detail"),
    path("runs/<uuid:pk>/stop/", views.stop_run, name="stop_run"),
    path("runs/<uuid:run_id>/results/", views.run_results, name="run_results"),
    path("runs/<uuid:run_id>/save/", views.run_save, name="run_save"),
    path("system/status/", views.system_status, name="system_status"),
    path("system/gpu/", views.gpu_info, name="gpu_info"),
    # Dashboard stats
    path("stats/", views.dashboard_stats, name="dashboard_stats"),
    path("health/", views.health_check, name="health_check"),
    # ── Attention Visualization Dashboard & Transformer Studios ──
    path("attention-visualization/canvas/", views_attention_dashboard.attention_canvas_endpoint, name="attn_canvas_api"),
    path("attention-visualization/interpretability/", views_attention_dashboard.interpretability_suite_endpoint, name="attn_interpretability_api"),
    path("attention-visualization/studios/representation/", views_attention_dashboard.representation_studios_endpoint, name="studio_representation_api"),
    path("attention-visualization/studios/computation/", views_attention_dashboard.computation_studios_endpoint, name="studio_computation_api"),
    path("attention-visualization/compare/", views_attention_dashboard.compare_architectures_endpoint, name="attn_compare_api"),
]
