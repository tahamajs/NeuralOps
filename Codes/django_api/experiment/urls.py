from api import views as api_views
from django.urls import path, re_path

from . import views

urlpatterns = [
    # Experiment CRUD
    path("experiments/", views.list_experiments, name="list_experiments"),
    path("experiments/create/", views.create_experiment, name="create_experiment"),
    path(
        "experiments/<uuid:exp_id>/", views.experiment_detail, name="experiment_detail"
    ),
    # Runs
    path("runs/", views.list_runs, name="list_runs"),
    path("runs/<uuid:run_id>/status/", views.job_status, name="job_status"),
    path(
        "runs/<uuid:run_id>/history/", views.training_history, name="training_history"
    ),
    path(
        "runs/<uuid:run_id>/snapshots/",
        views.training_snapshots,
        name="training_snapshots",
    ),
    path(
        "runs/<uuid:run_id>/config/", views.update_run_config, name="update_run_config"
    ),
    path("runs/<uuid:run_id>/artifacts/", views.run_artifacts, name="run_artifacts"),
    path(
        "runs/<uuid:run_id>/model/", views.run_model_download, name="run_model_download"
    ),
    re_path(
        r"^runs/(?P<run_id>[0-9a-f-]+)/artifacts/(?P<artifact_path>.+)$",
        views.run_artifact_download,
        name="run_artifact_download",
    ),
    path(
        "runs/<uuid:run_id>/analysis/",
        views.run_analysis_overview,
        name="run_analysis_overview",
    ),
    path(
        "runs/<uuid:run_id>/stability/",
        views.stability_analysis_view,
        name="stability_analysis",
    ),
    # Start jobs
    path("start_train/", views.start_train, name="start_train"),
    path("start_evaluate/<uuid:run_id>/", views.start_evaluate, name="start_evaluate"),
    path("start_compress/<uuid:run_id>/", views.start_compress, name="start_compress"),
    path("stop/<uuid:run_id>/", views.stop_job_endpoint, name="stop_job"),
    path("stop_all/", views.stop_all_jobs_endpoint, name="stop_all_jobs"),
    # Paper / report
    path("runs/<uuid:run_id>/paper/", views.generate_paper, name="generate_paper"),
    path(
        "runs/<uuid:run_id>/paper_preview/",
        views.paper_preview_pdf,
        name="paper_preview_pdf",
    ),
    path(
        "runs/<uuid:run_id>/report/", views.experiment_report, name="experiment_report"
    ),
    path("runs/<uuid:run_id>/editor/", views.paper_editor, name="paper_editor"),
    path(
        "runs/<uuid:run_id>/paper_source/",
        views.save_paper_source,
        name="save_paper_source",
    ),
    path("preview/<uuid:run_id>/", views.preview_paper, name="preview_paper"),
    # Comparison & diff
    path("compare/", views.compare_experiments, name="compare_experiments"),
    path("runs/<uuid:run_id_a>/diff/<uuid:run_id_b>/", views.run_diff, name="run_diff"),
    path("compare_runs/", views.compare_runs, name="compare_runs"),
    # Analytics
    path("analytics/hp-importance/", views.hp_importance_view, name="hp_importance"),
    path(
        "analytics/metric-correlation/",
        views.metric_correlation_view,
        name="metric_correlation",
    ),
    # Templates
    path("templates/", views.template_list_view, name="template_list"),
    path(
        "templates/<uuid:template_id>/",
        views.template_detail_view,
        name="template_detail",
    ),
    path(
        "templates/<uuid:template_id>/use/",
        views.use_template_view,
        name="use_template",
    ),
    # Config endpoints (static lists)
    path("config/attentions/", api_views.list_attentions, name="list_attentions"),
    path("config/models/", api_views.list_models, name="list_models"),
    path("config/tasks/", api_views.list_tasks, name="list_tasks"),
    path("config/compressors/", api_views.list_compressors, name="list_compressors"),
    path("config/methods/", api_views.list_methods, name="list_methods"),
    path("config/presets/", api_views.list_presets, name="list_presets"),
    # Auth (if you want separate auth endpoints)
    path("auth/register/", views.register_view, name="register"),
    path("auth/login/", views.login_view, name="login"),
    path("auth/logout/", views.logout_view, name="logout"),
    # System status
    path("system/status/", api_views.system_status, name="system_status"),
    path("system/gpu/", api_views.gpu_info, name="gpu_info"),
]
