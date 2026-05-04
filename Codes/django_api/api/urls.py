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

    # Experiments (containers)
    path("experiments/", views.list_experiments),
    path("experiments/create/", views.create_experiment),
    path("experiments/<uuid:exp_id>/", views.experiment_detail),

    # Runs
    path("runs/", views.list_runs),
    path("runs/<uuid:run_id>/", views.job_status, name="job_status"),
path("runs/<uuid:job_id>/", views.job_status, name="job_status"),
path("runs/<uuid:job_id>/stop/", views.stop_job, name="stop_job"),

    path("runs/exp/<uuid:exp_id>/", views.list_runs),
    path("runs/<uuid:run_id>/stop/", views.stop_job, name="stop_job"),
    path("runs/<uuid:run_id>/evaluate/", views.evaluate_job, name="evaluate_job"),
    path("runs/<uuid:run_id>/history/", views.training_history, name="training_history"),
    path("runs/<uuid:run_id>/snapshots/", views.training_snapshots, name="training_snapshots"),
    path("runs/<uuid:run_id>/config/", views.update_run_config, name="update_run_config"),
    path("runs/<uuid:run_id>/paper/", views.generate_paper, name="generate_paper"),
    path("runs/<uuid:run_id>/paper_preview/", views.paper_preview_pdf, name="paper_preview_pdf"),
    path("runs/<uuid:run_id>/report/", views.experiment_report, name="experiment_report"),
    path("runs/<uuid:run_id>/resume/", views.resume_training, name="resume_training"),
    path("runs/<uuid:run_id>/editor/", views.paper_editor, name="paper_editor"),
    path("runs/<uuid:run_id>/paper_source/", views.save_paper_source, name="save_paper_source"),
    path("runs/<uuid:run_id>/artifacts/", views.run_artifacts, name="run_artifacts"),
    path("runs/<uuid:run_id>/model/", views.run_model_download, name="run_model_download"),
    re_path(r"runs/(?P<run_id>[0-9a-f-]+)/artifacts/(?P<artifact_path>.+)$", views.run_artifact_download, name="run_artifact_download"),

    # Start jobs (create runs)
    path("train/", views.start_train),
    path("evaluate/", views.start_evaluate),
    path("compress/", views.start_compress),

    # System & GPU
    path("system/status/", views.system_status),
    path("system/gpu/", views.gpu_info),

    # Authentication
    path("auth/register/", views.register_view),
    path("auth/login/", views.login_view),
    path("auth/logout/", views.logout_view),

    # Preview (HTML modal)
    path("preview/<uuid:run_id>/", views.preview_paper),

    # Stop all jobs
    path("jobs/stop_all/", views.stop_all_jobs, name="stop_all_jobs"),
]