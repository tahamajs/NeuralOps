from django.urls import path
from . import views

urlpatterns = [
    # Projects
    path('projects/', views.project_list, name='impact_project_list'),
    path('projects/<uuid:project_id>/', views.project_detail, name='impact_project_detail'),
    path('projects/<uuid:project_id>/rescore/', views.rescore_project, name='impact_project_rescore'),

    # Predict (without saving)
    path('predict/', views.predict_impact, name='predict_impact'),

    # Leaderboard
    path('leaderboard/', views.project_leaderboard, name='impact_leaderboard'),
]