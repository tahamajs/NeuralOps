from django.urls import path

from . import views

urlpatterns = [
    path("", views.project_list, name="lab_assistant_root"),
    # Projects
    path("projects/", views.project_list, name="annotation_project_list"),
    path(
        "projects/<uuid:project_id>/",
        views.project_detail,
        name="annotation_project_detail",
    ),
    # Tasks
    path(
        "projects/<uuid:project_id>/tasks/",
        views.task_list,
        name="annotation_task_list",
    ),
    path(
        "projects/<uuid:project_id>/tasks/<uuid:task_id>/",
        views.task_detail,
        name="annotation_task_detail",
    ),
    # AI Pre-labeling
    path(
        "projects/<uuid:project_id>/ai-label/",
        views.run_ai_pre_labeling,
        name="annotation_ai_label",
    ),
    # Reviews
    path(
        "projects/<uuid:project_id>/tasks/<uuid:task_id>/review/",
        views.submit_review,
        name="annotation_review",
    ),
    # Guidelines
    path(
        "projects/<uuid:project_id>/guidelines/",
        views.guidelines,
        name="annotation_guidelines",
    ),
    # Label Studio Sync
    path(
        "projects/<uuid:project_id>/sync-label-studio/",
        views.sync_to_label_studio,
        name="annotation_sync_ls",
    ),
    # AI Feedback
    path(
        "projects/<uuid:project_id>/tasks/<uuid:task_id>/ai-feedback/",
        views.submit_ai_feedback,
        name="annotation_ai_feedback",
    ),
]
