from django.urls import path, re_path
from . import views

urlpatterns = [
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
    re_path(r"^runs/(?P<run_id>[0-9a-f-]+)/artifacts/(?P<artifact_path>.+)$", views.run_artifact_download, name="run_artifact_download"),
    path("runs/stop_all/", views.stop_all_jobs_endpoint, name="stop_all_jobs"),
    path("preview/<uuid:run_id>/", views.preview_paper, name="preview_paper"),

    path("runs/<uuid:run_id>/analysis/", views.run_analysis_overview, name="run_analysis_overview"),

]