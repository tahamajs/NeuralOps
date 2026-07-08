from django.urls import path

from . import views

urlpatterns = [
    # Analysis
    path("analysis/", views.analysis_list, name="analysis_list"),
    path(
        "analysis/<uuid:analysis_id>/read/",
        views.mark_analysis_read,
        name="mark_analysis_read",
    ),
    path(
        "analysis/<uuid:analysis_id>/archive/",
        views.archive_analysis,
        name="archive_analysis",
    ),
    path("analysis/trigger/", views.trigger_analysis, name="trigger_analysis"),
    # Recommendations
    path("recommendations/", views.recommendation_list, name="recommendation_list"),
    path(
        "recommendations/<uuid:recommendation_id>/apply/",
        views.apply_recommendation,
        name="apply_recommendation",
    ),
    path(
        "recommendations/<uuid:recommendation_id>/dismiss/",
        views.dismiss_recommendation,
        name="dismiss_recommendation",
    ),
    # Chat
    path("chat/", views.chat_session_list, name="chat_session_list"),
    path(
        "chat/<uuid:session_id>/", views.chat_session_detail, name="chat_session_detail"
    ),
    path(
        "chat/<uuid:session_id>/delete/",
        views.chat_session_delete,
        name="chat_session_delete",
    ),
]
