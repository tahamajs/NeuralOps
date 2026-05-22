from django.urls import path
from . import views

urlpatterns = [
    path('publications/', views.publication_list, name='publication_list'),
    path('publications/<uuid:pub_id>/', views.publication_detail, name='publication_detail'),
    path('citations/', views.add_citation, name='add_citation'),
    path('publications/<uuid:pub_id>/reviews/', views.review_list, name='review_list'),
]