from django.urls import path, re_path
from . import views

urlpatterns = [
    # Config
    path("config/attentions/", views.list_attentions),
    path("config/models/", views.list_models),
    path("config/tasks/", views.list_tasks),
    path("config/compressors/", views.list_compressors),
    path("config/methods/", views.list_methods),
    path("config/presets/", views.list_presets),

    # Experiments
    path("experiments/", views.list_experiments),
    path("experiments/create/", views.create_experiment),
    path("experiments/<uuid:exp_id>/", views.experiment_detail),

    # Runs
    path("runs/", views.list_runs),
    path("runs/<uuid:run_id>/", views.job_status, name="job_status"),
    path("runs/<uuid:run_id>/stop/", views.stop_job_endpoint, name="stop_job"),
    path("runs/exp/<uuid:exp_id>/", views.list_runs),
    path("runs/<uuid:run_id>/evaluate/", views.start_evaluate, name="evaluate_job"),
    path("runs/<uuid:run_id>/history/", views.training_history),
    path("runs/<uuid:run_id>/snapshots/", views.training_snapshots),
    path("runs/<uuid:run_id>/config/", views.update_run_config),
    path("runs/<uuid:run_id>/paper/", views.generate_paper),
    path("runs/<uuid:run_id>/paper_preview/", views.paper_preview_pdf),
    path("runs/<uuid:run_id>/report/", views.experiment_report),
    path("runs/<uuid:run_id>/editor/", views.paper_editor),
    path("runs/<uuid:run_id>/paper_source/", views.save_paper_source),
    path("runs/<uuid:run_id>/artifacts/", views.run_artifacts),
    path("runs/<uuid:run_id>/model/", views.run_model_download),
    re_path(r"runs/(?P<run_id>[0-9a-f-]+)/artifacts/(?P<artifact_path>.+)$", views.run_artifact_download),

    # Start jobs
    path("train/", views.start_train),
    path("evaluate/", views.start_evaluate),
    path("compress/", views.start_compress),

    # System
    path("system/status/", views.system_status),
    path("system/gpu/", views.gpu_info),

    # Auth
    path("auth/register/", views.register_view),
    path("auth/login/", views.login_view),
    path("auth/logout/", views.logout_view),

    # Preview
    path("preview/<uuid:run_id>/", views.preview_paper),

    # Stop all
    path("jobs/stop_all/", views.stop_all_jobs_endpoint),
]