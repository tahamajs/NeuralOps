# literature_review/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("", views.project_list, name="literature_review_root"),
    path("projects/", views.project_list, name="lit_project_list"),
    path(
        "projects/<uuid:project_id>/", views.project_detail, name="lit_project_detail"
    ),
    path(
        "projects/<uuid:project_id>/topic-finder/",
        views.start_topic_finder,
        name="topic_finder",
    ),
    path(
        "projects/<uuid:project_id>/notebook/",
        views.generate_notebook,
        name="generate_notebook",
    ),
    path(
        "projects/<uuid:project_id>/deep-review/",
        views.start_deep_review,
        name="deep_review",
    ),
    path(
        "projects/<uuid:project_id>/sections/",
        views.notebook_section_list,
        name="notebook_sections",
    ),
    path(
        "sections/<uuid:section_id>/",
        views.notebook_section_detail,
        name="notebook_section_detail",
    ),
    path(
        "projects/<uuid:project_id>/reviews/",
        views.deep_review_list,
        name="deep_reviews",
    ),
    path(
        "reviews/<uuid:review_id>/", views.deep_review_detail, name="deep_review_detail"
    ),
]
