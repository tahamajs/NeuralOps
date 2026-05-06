# experiments/urls.py
from django.urls import path, re_path
from . import views

urlpatterns = [
    # ------------------------------------------------------------------
    # Configuration / Registry endpoints (view‑only)
    # ------------------------------------------------------------------
    path("config/attentions/", views.list_attentions, name="list_attentions"),
    path("config/models/", views.list_models, name="list_models"),
    path("config/tasks/", views.list_tasks, name="list_tasks"),
    path("config/compressors/", views.list_compressors, name="list_compressors"),
    path("config/methods/", views.list_methods, name="list_methods"),
    path("config/presets/", views.list_presets, name="list_presets"),

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------
    path("experiments/", views.list_experiments, name="list_experiments"),
    path("experiments/create/", views.create_experiment, name="create_experiment"),
    path("experiments/<uuid:exp_id>/", views.experiment_detail, name="experiment_detail"),

    # ------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------
    path("runs/", views.list_runs, name="list_runs"),
    path("runs/<uuid:run_id>/", views.job_status, name="job_status"),
    path("runs/<uuid:run_id>/stop/", views.stop_job_endpoint, name="stop_job"),
    path("runs/exp/<uuid:exp_id>/", views.list_runs, name="runs_by_experiment"),
    path("runs/<uuid:run_id>/evaluate/", views.start_evaluate, name="evaluate_job"),
    path("runs/<uuid:run_id>/history/", views.training_history, name="training_history"),
    path("runs/<uuid:run_id>/snapshots/", views.training_snapshots, name="training_snapshots"),
    path("runs/<uuid:run_id>/config/", views.update_run_config, name="update_run_config"),
    path("runs/<uuid:run_id>/paper/", views.generate_paper, name="generate_paper"),
    path("runs/<uuid:run_id>/paper_preview/", views.paper_preview_pdf, name="paper_preview_pdf"),
    path("runs/<uuid:run_id>/report/", views.experiment_report, name="experiment_report"),
    path("runs/<uuid:run_id>/editor/", views.paper_editor, name="paper_editor"),
    path("runs/<uuid:run_id>/paper_source/", views.save_paper_source, name="save_paper_source"),
    path("runs/<uuid:run_id>/artifacts/", views.run_artifacts, name="run_artifacts"),
    path("runs/<uuid:run_id>/model/", views.run_model_download, name="run_model_download"),
    # Catch‑all for downloading individual artifacts (supports sub‑paths)
    re_path(
        r"^runs/(?P<run_id>[0-9a-f-]+)/artifacts/(?P<artifact_path>.+)$",
        views.run_artifact_download,
        name="run_artifact_download",
    ),

    # ------------------------------------------------------------------
    # Job start endpoints (body contains details)
    # ------------------------------------------------------------------
    path("train/", views.start_train, name="start_train"),
    path("evaluate/", views.start_evaluate, name="start_evaluate"),
    path("compress/", views.start_compress, name="start_compress"),

    # ------------------------------------------------------------------
    # System status & GPU info (public / no auth required)
    # ------------------------------------------------------------------
    path("system/status/", views.system_status, name="system_status"),
    path("system/gpu/", views.gpu_info, name="gpu_info"),

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    path("auth/register/", views.register_view, name="register"),
    path("auth/login/", views.login_view, name="login"),
    path("auth/logout/", views.logout_view, name="logout"),

    # ------------------------------------------------------------------
    # Paper preview (may be used by dashboard)
    # ------------------------------------------------------------------
    path("preview/<uuid:run_id>/", views.preview_paper, name="preview_paper"),

    # ------------------------------------------------------------------
    # Administrative: stop all running jobs
    # ------------------------------------------------------------------
    path("jobs/stop_all/", views.stop_all_jobs_endpoint, name="stop_all_jobs"),
]