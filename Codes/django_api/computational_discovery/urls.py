from django.urls import path

from . import views

urlpatterns = [
    path("", views.experiment_list, name="computational_discovery_root"),
    # Experiments
    path("experiments/", views.experiment_list, name="code_experiment_list"),
    path(
        "experiments/<uuid:experiment_id>/",
        views.experiment_detail,
        name="code_experiment_detail",
    ),
    path(
        "experiments/<uuid:experiment_id>/launch/",
        views.launch_experiment,
        name="launch_experiment",
    ),
    path(
        "experiments/<uuid:experiment_id>/pause/",
        views.pause_experiment,
        name="pause_experiment",
    ),
    # Variants
    path(
        "experiments/<uuid:experiment_id>/variants/",
        views.variant_list,
        name="variant_list",
    ),
    path("variants/<uuid:variant_id>/", views.variant_detail, name="variant_detail"),
    # Runs
    path("variants/<uuid:variant_id>/runs/", views.run_list, name="run_list"),
    path("runs/<uuid:run_id>/", views.run_detail, name="run_detail"),
    # Leaderboard
    path("leaderboard/", views.global_leaderboard, name="code_leaderboard"),
]
