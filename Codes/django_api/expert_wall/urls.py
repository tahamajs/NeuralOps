from django.urls import path
from . import views

urlpatterns = [
    # Expert Profiles
    path('profile/', views.my_expert_profile, name='expert_profile'),
    path('profile/regenerate/', views.regenerate_expert_profile, name='regenerate_profile'),
    path('profile/<int:user_id>/', views.expert_profile_detail, name='expert_profile_detail'),
    path('directory/', views.expert_directory, name='expert_directory'),

    # Projects
    path('projects/', views.project_list, name='project_list'),
    path('projects/<uuid:project_id>/', views.project_detail, name='project_detail'),
    path('projects/<uuid:project_id>/apply/', views.apply_to_project, name='apply_to_project'),
    path('projects/<uuid:project_id>/applications/<uuid:application_id>/', views.handle_application, name='handle_application'),

    # Expert Reviews
    path('reviews/', views.review_list, name='review_list'),
    path('reviews/open/', views.review_requests_for_me, name='review_requests_for_me'),
    path('reviews/<uuid:review_id>/accept/', views.accept_review, name='accept_review'),
    path('reviews/<uuid:review_id>/complete/', views.complete_review, name='complete_review'),

    # Matching
    path('projects/<uuid:project_id>/matched-experts/', views.matched_experts_for_project, name='matched_experts_for_project'),
]