from django.urls import path

from . import views

urlpatterns = [
    path("", views.tournament_list, name="idea_tournament_root"),
    path("tournaments/", views.tournament_list, name="tournament_list"),
    path(
        "tournaments/<uuid:tournament_id>/",
        views.tournament_detail,
        name="tournament_detail",
    ),
    path(
        "tournaments/<uuid:tournament_id>/start/",
        views.start_tournament,
        name="start_tournament",
    ),
    path(
        "tournaments/<uuid:tournament_id>/hypotheses/",
        views.hypothesis_list,
        name="hypothesis_list",
    ),
    path(
        "tournaments/<uuid:tournament_id>/winners/",
        views.tournament_winners,
        name="tournament_winners",
    ),
    path(
        "hypotheses/<uuid:hypothesis_id>/",
        views.hypothesis_detail,
        name="hypothesis_detail",
    ),
]
