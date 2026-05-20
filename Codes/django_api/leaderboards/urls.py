# leaderboards/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # ── Benchmarks ───────────────────────────────────────────
    path('benchmarks/', views.benchmark_list, name='benchmark_list'),
    path('benchmarks/<uuid:benchmark_id>/', views.benchmark_detail, name='benchmark_detail'),
    path('benchmarks/<uuid:benchmark_id>/leaderboard/', views.leaderboard_view, name='leaderboard'),
    path('benchmarks/<uuid:benchmark_id>/my-rank/', views.my_rank, name='my_rank'),

    # ── Submissions ──────────────────────────────────────────
    path('benchmarks/<uuid:benchmark_id>/submissions/', views.submission_list, name='submission_list'),
    path('benchmarks/<uuid:benchmark_id>/submissions/<uuid:submission_id>/', views.submission_detail, name='submission_detail'),

    # ── Voting ──────────────────────────────────────────────
    path('benchmarks/<uuid:benchmark_id>/submissions/<uuid:submission_id>/vote/', views.vote_submission, name='vote'),
    path('benchmarks/<uuid:benchmark_id>/submissions/<uuid:submission_id>/vote/remove/', views.remove_vote, name='remove_vote'),

    # ── Badges ──────────────────────────────────────────────
    path('badges/', views.badge_list, name='badge_list'),
    path('badges/mine/', views.my_badges, name='my_badges'),

    # ── Follow ──────────────────────────────────────────────
    path('benchmarks/<uuid:benchmark_id>/follow/', views.follow_benchmark, name='follow'),
    path('benchmarks/<uuid:benchmark_id>/unfollow/', views.unfollow_benchmark, name='unfollow'),

    # ── Reference data ──────────────────────────────────────
    path('tasks/', views.task_list, name='task_list'),
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('metrics/', views.metric_list, name='metric_list'),

    # ── Global stats ────────────────────────────────────────
    path('stats/', views.global_stats, name='global_stats'),
]