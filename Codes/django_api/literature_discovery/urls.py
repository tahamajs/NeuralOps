from django.urls import path
from . import views

urlpatterns = [
    # Citation Graph
    path('graph/<uuid:paper_id>/', views.paper_graph, name='paper_graph'),
    path('graph/<uuid:paper_id>/visualise/', views.paper_graph_visualisation, name='paper_graph_visualise'),

    # Pathfinding
    path('find-path/', views.find_path, name='find_citation_path'),
    path('paths/', views.saved_paths, name='saved_paths'),

    # Recommendations
    path('recommendations/', views.recommendations, name='paper_recommendations'),
    path('recommendations/generate/', views.generate_recommendations, name='generate_recommendations'),
    path('recommendations/<uuid:rec_id>/dismiss/', views.dismiss_recommendation, name='dismiss_recommendation'),
    path('recommendations/<uuid:rec_id>/save/', views.save_recommendation, name='save_recommendation'),

    # Landscapes
    path('landscapes/', views.landscape_list, name='landscape_list'),
    path('landscapes/<uuid:landscape_id>/', views.landscape_detail, name='landscape_detail'),
    path('landscapes/generate/', views.generate_landscape, name='generate_landscape'),

    # Feed
    path('feed/', views.paper_feed, name='paper_feed'),
    path('feed/<uuid:feed_id>/seen/', views.mark_feed_seen, name='mark_feed_seen'),
    path('feed/<uuid:feed_id>/clicked/', views.mark_feed_clicked, name='mark_feed_clicked'),
]