from django.urls import path

from . import views

urlpatterns = [
    # Failure Analysis
    path("analyze/<uuid:run_id>/", views.analyze_failure, name="analyze_failure"),
    path("analyses/<uuid:analysis_id>/", views.analysis_detail, name="analysis_detail"),
    path("analyses/", views.my_analyses, name="my_analyses"),
    path("runs/<uuid:run_id>/analysis/", views.run_analysis, name="run_analysis"),
    # Fix Actions
    path("fixes/<uuid:fix_id>/apply/", views.apply_fix, name="apply_fix"),
    # Error Patterns
    path("patterns/", views.error_pattern_list, name="error_pattern_list"),
    path("patterns/create/", views.error_pattern_create, name="error_pattern_create"),
    # Stats
    path("stats/", views.failure_stats, name="failure_stats"),
]
