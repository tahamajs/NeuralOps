from django.urls import path

from . import views

urlpatterns = [
    path("", views.session_list, name="ai_coscientist_root"),
    path("start/", views.create_session, name="session_start"),
    path("sessions/", views.session_list, name="session_list"),
    path("sessions/create/", views.create_session, name="session_create"),
    path("sessions/<uuid:session_id>/", views.session_detail, name="session_detail"),
    path(
        "sessions/<uuid:session_id>/hypotheses/",
        views.session_hypotheses,
        name="session_hypotheses",
    ),
    path(
        "sessions/<uuid:session_id>/stop/",
        views.session_stop,
        name="session_stop",
    ),
    path(
        "sessions/<uuid:session_id>/evolve/",
        views.session_evolve,
        name="session_evolve",
    ),
    path(
        "sessions/<uuid:session_id>/tournament/",
        views.tournament_history,
        name="tournament_history",
    ),
    path(
        "hypotheses/<uuid:hypothesis_id>/",
        views.hypothesis_detail,
        name="hypothesis_detail",
    ),
]
