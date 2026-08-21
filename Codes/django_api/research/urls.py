from django.urls import path

from . import views, views_stream

urlpatterns = [
    path("", views.publication_list, name="research_root"),
    path("publications/", views.publication_list, name="publication_list"),
    path(
        "publications/<uuid:pub_id>/",
        views.publication_detail,
        name="publication_detail",
    ),
    path("citations/", views.add_citation, name="add_citation"),
    path("publications/<uuid:pub_id>/reviews/", views.review_list, name="review_list"),
]
path("research-stream/", views_stream.get_research_stream),
