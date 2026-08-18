from django.urls import path
from . import views

urlpatterns = [
    path("", views.my_insights, name="experiment_accelerator_root"),
    # KPIs & Opportunities
    path("kpis/", views.kpi_metrics, name="kpi_metrics"),
    path("opportunity/", views.opportunity_score, name="opportunity_score"),
    # Insights
    path("insights/", views.my_insights, name="my_insights"),
    path("insights/generate/", views.generate_insights, name="generate_insights"),
    path("insights/<str:insight_id>/read/", views.mark_insight_read, name="mark_insight_read"),
    path("insights/<str:insight_id>/dismiss/", views.dismiss_insight, name="dismiss_insight"),
    path("insights/mark-all-read/", views.mark_all_insights_read, name="mark_all_insights_read"),
    # Comparisons & Interventions
    path("compare/", views.compare_experiments, name="compare_experiments"),
    path("interventions/", views.interventions_list, name="interventions_list"),
    path("interventions/<str:intervention_id>/apply/", views.apply_intervention, name="apply_intervention"),
    path("auto-refresh/", views.auto_refresh_config, name="auto_refresh_config"),
    # Recommendations
    path("recommendations/", views.recommendations, name="recommendations"),
    path("recommendations/<str:recommendation_id>/launch/", views.launch_recommendation, name="launch_recommendation"),
    path("recommendations/<str:recommendation_id>/dismiss/", views.dismiss_recommendation, name="dismiss_recommendation"),
    # Summaries
    path("summaries/", views.summaries, name="summaries"),
]
