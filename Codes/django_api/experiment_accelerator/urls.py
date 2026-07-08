from django.urls import path

from . import views

urlpatterns = [
    # Insights
    path("insights/", views.my_insights, name="my_insights"),
    path("insights/generate/", views.generate_insights, name="generate_insights"),
    path(
        "insights/<uuid:insight_id>/read/",
        views.mark_insight_read,
        name="mark_insight_read",
    ),
    path(
        "insights/mark-all-read/",
        views.mark_all_insights_read,
        name="mark_all_insights_read",
    ),
    # Recommendations
    path("recommendations/", views.recommendations, name="recommendations"),
    path(
        "recommendations/<uuid:recommendation_id>/launch/",
        views.launch_recommendation,
        name="launch_recommendation",
    ),
    path(
        "recommendations/<uuid:recommendation_id>/dismiss/",
        views.dismiss_recommendation,
        name="dismiss_recommendation",
    ),
    # Summaries
    path("summaries/", views.summaries, name="summaries"),
]
