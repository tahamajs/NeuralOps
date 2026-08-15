from django.urls import path

from . import views

urlpatterns = [
    path("", views.generate_hypotheses, name="research_discovery_root"),
    path(
        "literature/synthesize/",
        views.synthesize_literature,
        name="synthesize_literature",
    ),
    path("hypotheses/generate/", views.generate_hypotheses, name="generate_hypotheses"),
    path("databases/", views.list_databases, name="list_databases"),
    path(
        "databases/<str:db_key>/search/", views.search_database, name="search_database"
    ),
    path(
        "databases/cross-search/",
        views.cross_database_search,
        name="cross_database_search",
    ),
    path("gaps/analyze/", views.analyze_research_gaps, name="analyze_research_gaps"),
    path("grants/search/", views.search_grants, name="search_grants"),
    path("grants/match/", views.match_grants, name="match_grants"),
]
