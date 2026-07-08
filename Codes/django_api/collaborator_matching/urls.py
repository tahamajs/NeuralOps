from django.urls import path

from . import views

urlpatterns = [
    # Interest Profile
    path("profile/", views.my_interest_profile, name="interest_profile"),
    path(
        "profile/regenerate/",
        views.regenerate_interest_profile,
        name="regenerate_profile",
    ),
    # Matches
    path("suggestions/", views.suggested_matches, name="suggested_matches"),
    path(
        "suggestions/<uuid:match_id>/accept/", views.accept_match, name="accept_match"
    ),
    path(
        "suggestions/<uuid:match_id>/dismiss/",
        views.dismiss_match,
        name="dismiss_match",
    ),
    path("history/", views.match_history, name="match_history"),
    # Interdisciplinary
    path(
        "interdisciplinary/",
        views.interdisciplinary_matches,
        name="interdisciplinary_matches",
    ),
]
