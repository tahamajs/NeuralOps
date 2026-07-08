from django.urls import path

from . import views

urlpatterns = [
    # Weights
    path("weights/", views.get_weights, name="reputation_weights"),
    path("weights/update/", views.update_weights, name="reputation_weights_update"),
    # Scores & Leaderboard
    path("me/", views.my_reputation, name="my_reputation"),
    path("user/<str:username>/", views.user_reputation, name="user_reputation"),
    path("leaderboard/", views.reputation_leaderboard, name="reputation_leaderboard"),
    # Events
    path("events/", views.my_events, name="reputation_events"),
    path("events/record/", views.record_event, name="reputation_event_record"),
    # Recommendations
    path(
        "recommendations/", views.my_recommendations, name="reputation_recommendations"
    ),
    path(
        "recommendations/generate/",
        views.generate_recommendations,
        name="reputation_generate_recommendations",
    ),
    path(
        "recommendations/<uuid:recommendation_id>/dismiss/",
        views.dismiss_recommendation,
        name="reputation_dismiss_recommendation",
    ),
    # Snapshots
    path("snapshots/", views.my_snapshots, name="reputation_snapshots"),
    path("snapshots/take/", views.take_snapshot, name="reputation_take_snapshot"),
]
