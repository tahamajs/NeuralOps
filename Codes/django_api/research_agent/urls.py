from django.urls import path
from . import views

urlpatterns = [
    path('sessions/', views.session_list, name='agent_session_list'),
    path('sessions/<uuid:session_id>/', views.session_detail, name='agent_session_detail'),
    path('ideas/generate/', views.generate_ideas, name='generate_ideas'),
    path('ideas/', views.my_ideas, name='my_ideas'),
    path('ideas/<uuid:idea_id>/save/', views.save_idea, name='save_idea'),
]