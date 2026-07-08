from api import views
from django.urls import include, path

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
    # Agent Chat
    path("chats/", views.ChatSessionListCreateView.as_view(), name="chat_sessions"),
    path(
        "chats/<uuid:session_id>/",
        views.ChatSessionDetailView.as_view(),
        name="chat_session_detail",
    ),
    path(
        "chats/<uuid:session_id>/messages/",
        views.MessageListView.as_view(),
        name="chat_messages",
    ),
    path("chats/<uuid:session_id>/send/", views.send_message, name="send_message"),
    path(
        "chats/<uuid:session_id>/summary/",
        views.api_conversation_summary,
        name="conversation_summary",
    ),
    path(
        "chats/<uuid:session_id>/export/",
        views.api_export_conversation,
        name="export_conversation",
    ),
    # Paper Research
    path("papers/search/", views.api_search_papers, name="search_papers"),
    path("papers/save/", views.api_save_paper, name="save_paper"),
    path("papers/saved/", views.api_saved_papers_list, name="saved_papers"),
    path("papers/saved/<uuid:paper_id>/", views.api_paper_detail, name="paper_detail"),
    path(
        "papers/saved/<uuid:paper_id>/annotate/",
        views.api_annotate_paper,
        name="annotate_paper",
    ),
    path(
        "papers/saved/<uuid:paper_id>/delete/",
        views.api_delete_paper,
        name="delete_paper",
    ),
    path("papers/library/", views.api_search_library, name="search_library"),
    path("papers/citation-graph/", views.api_citation_graph, name="citation_graph"),
    path(
        "papers/bulk-import/", views.api_bulk_import_papers, name="bulk_import_papers"
    ),
    path(
        "papers/recommendations/<uuid:run_id>/",
        views.paper_recommendations_for_run,
        name="paper_recommendations_for_run",
    ),
    path(
        "papers/recommendations/library/",
        views.paper_recommendations_from_library,
        name="paper_recommendations_library",
    ),
    path(
        "papers/search-for-run/<uuid:run_id>/",
        views.search_papers_for_run_topic,
        name="search_papers_for_run",
    ),
    # Training Analysis
    path("training/analyze/", views.api_analyze_run, name="analyze_run"),
    path(
        "training/analysis/<uuid:run_id>/", views.api_run_analysis, name="run_analysis"
    ),
    path("training/suggest-next/", views.api_suggest_next, name="suggest_next"),
    # Monitor
    path("monitor/start/", views.start_training_monitor, name="start_monitor"),
    # Prompt Testing
    path("prompts/test/", views.api_test_prompt, name="test_prompt"),
    # Experiments
    path("experiments/", views.experiment_list, name="experiment_list"),
    path("experiments/<uuid:pk>/", views.experiment_detail, name="experiment_detail"),
    # Runs
    path("runs/start-train/", views.start_train, name="start_train"),
    path("runs/start-evaluate/", views.start_evaluate, name="start_evaluate"),
    path("runs/start-compress/", views.start_compress, name="start_compress"),
    path("runs/<uuid:pk>/", views.run_detail, name="run_detail"),
    path("runs/<uuid:pk>/stop/", views.stop_run, name="stop_run"),
    path("runs/<uuid:run_id>/results/", views.run_results, name="run_results"),
    path("runs/<uuid:run_id>/save/", views.run_save, name="run_save"),
    # System
    path("health/", views.health_check, name="health_check"),
    path("version/", views.version, name="version"),
    path("system/status/", views.system_status, name="system_status"),
    path("system/gpu/", views.gpu_info, name="gpu_info"),
    # Search & Activity
    path("search/", views.global_search, name="global_search"),
    path("activity/", views.activity_feed, name="activity_feed"),
    # Sub‑apps
    path("", include("experiment.urls")),
    path("", include("jobscheduler.urls")),
    path("", include("notes.urls")),
]
