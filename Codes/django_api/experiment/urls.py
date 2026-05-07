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
    path('compare/', views.compare_experiments, name='compare_experiments'),
    path('runs/<uuid:run_id_a>/diff/<uuid:run_id_b>/', views.run_diff, name='run_diff'),
    path('analytics/hp-importance/', views.hp_importance_view, name='hp_importance'),
    path('analytics/metric-correlation/', views.metric_correlation_view, name='metric_correlation'),
    path('runs/<uuid:run_id>/stability/', views.stability_analysis_view, name='stability_analysis'),
    path('templates/', views.template_list, name='template_list'),
    path('templates/<uuid:template_id>/', views.template_detail, name='template_detail'),
    path('templates/<uuid:template_id>/use/', views.use_template, name='use_template'),

]