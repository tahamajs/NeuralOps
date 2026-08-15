from django.urls import path

from . import views

urlpatterns = [
    path("", views.template_list, name="experiment_recommender_root"),
    # Templates
    path("templates/", views.template_list, name="recommender_templates"),
    path("templates/<uuid:template_id>/use/", views.use_template, name="use_template"),
    # Opportunity Index
    path("opportunity/", views.opportunity_index, name="opportunity_index"),
    path(
        "opportunity/recalculate/",
        views.recalculate_opportunity,
        name="recalculate_opportunity",
    ),
    # Recommendations
    path("suggestions/", views.recommendation_list, name="recommendation_list"),
    path(
        "suggestions/<uuid:recommendation_id>/accept/",
        views.accept_recommendation,
        name="accept_recommendation",
    ),
    path(
        "suggestions/<uuid:recommendation_id>/dismiss/",
        views.dismiss_recommendation,
        name="dismiss_recommendation",
    ),
    path(
        "suggestions/regenerate/",
        views.regenerate_recommendations,
        name="regenerate_recommendations",
    ),
    # Creative Variants
    path("variants/<uuid:run_id>/", views.creative_variants, name="creative_variants"),
    path(
        "variants/accept/<uuid:variant_id>/",
        views.accept_variant,
        name="accept_variant",
    ),
]
