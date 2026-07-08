from django.urls import path

from . import views

urlpatterns = [
    # ── Benchmark Suites & Tests ──────────────────────────────
    path("suites/", views.suite_list, name="benchmark_suite_list"),
    path("suites/<uuid:suite_id>/", views.suite_detail, name="benchmark_suite_detail"),
    path("tests/", views.test_list, name="benchmark_test_list"),
    # ── Benchmark Runs ────────────────────────────────────────
    path("runs/", views.run_list, name="benchmark_run_list"),
    path("runs/<uuid:run_id>/", views.run_detail, name="benchmark_run_detail"),
    path(
        "listings/<uuid:listing_id>/run/", views.start_benchmark, name="start_benchmark"
    ),
    # ── Scores & Leaderboard ──────────────────────────────────
    path("scores/", views.score_list, name="benchmark_score_list"),
    path(
        "scores/<uuid:listing_id>/", views.score_detail, name="benchmark_score_detail"
    ),
    path(
        "scores/<uuid:listing_id>/recalculate/",
        views.recalculate_score,
        name="recalculate_score",
    ),
    path("leaderboard/", views.gpu_leaderboard, name="gpu_leaderboard"),
    # ── Provider Dashboard ───────────────────────────────────
    path("dashboard/", views.my_benchmark_dashboard, name="my_benchmark_dashboard"),
]
