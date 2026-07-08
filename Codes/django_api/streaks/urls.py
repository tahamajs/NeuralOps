from django.urls import path

from . import views

urlpatterns = [
    # Heatmap
    path("heatmap/<str:username>/", views.heatmap_data, name="heatmap_data"),
    # Streaks
    path("streak/<str:username>/", views.streak_info, name="streak_info"),
    path("streak/me/", views.my_streak, name="my_streak"),
    path("leaderboard/", views.streak_leaderboard, name="streak_leaderboard"),
    # Weekly Summaries
    path("weekly/me/", views.my_weekly_summaries, name="my_weekly_summaries"),
    path("weekly/<str:username>/", views.weekly_summaries, name="weekly_summaries"),
    path(
        "weekly/generate/",
        views.generate_weekly_summary,
        name="generate_weekly_summary",
    ),
    # Daily Activities
    path("daily/me/", views.my_daily_activities, name="my_daily_activities"),
]
