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

    # Jobs
    path("train/", views.start_train),
    path("evaluate/", views.start_evaluate),
    path("compress/", views.start_compress),
    path("jobs/<uuid:job_id>/", views.job_status),
    path("jobs/<uuid:job_id>/stop/", views.stop_job),
    path("jobs/<uuid:job_id>/config/", views.update_job_config),

    # Experiments
    path("experiments/", views.list_experiments),
    path("experiments/<uuid:job_id>/artifacts/", views.experiment_artifacts),
    re_path(r"experiments/(?P<job_id>[0-9a-f-]+)/artifacts/(?P<artifact_path>.+)$", views.experiment_artifact_download),
    path("experiments/<uuid:job_id>/model/", views.experiment_model_download),

    # System & Paper
    path("experiments/<uuid:job_id>/paper/", views.generate_paper),
    path("experiments/<str:job_id>/paper_preview/", views.paper_preview_pdf, name="paper-preview"),
    path("experiments/<str:job_id>/report/", views.experiment_report, name="experiment-report"),
    path("system/status/", views.system_status),
    path("system/gpu/", views.gpu_info),

    # Authentication (views are inside api/views.py)
    path("auth/register/", views.register_view),
    path("auth/login/", views.login_view),
    path("auth/logout/", views.logout_view),
    path('jobs/<str:job_id>/evaluate/', views.evaluate_job, name='evaluate-job'),
]