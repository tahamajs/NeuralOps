from django.urls import path
from . import views

urlpatterns = [
    path('chats/', views.ChatSessionList.as_view(), name='chat_sessions'),
    path('chats/<uuid:session_id>/', views.ChatSessionDetail.as_view(), name='chat_session'),
    path('chats/<uuid:session_id>/messages/', views.MessageList.as_view(), name='chat_messages'),
]

from django.urls import path
from . import views

urlpatterns = [
    path('chats/', views.ChatSessionListCreateView.as_view(), name='chat_sessions'),
    path('chats/<uuid:session_id>/', views.ChatSessionDetailView.as_view(), name='chat_session_detail'),
    path('chats/<uuid:session_id>/messages/', views.MessageListView.as_view(), name='chat_messages'),
    path('chats/<uuid:session_id>/send/', views.send_message, name='send_message'),
    path('papers/search/', views.api_search_papers, name='search_papers'),
path('papers/save/', views.api_save_paper, name='save_paper'),
path('papers/library/', views.api_search_library, name='search_library'),
path('papers/citation-graph/', views.api_citation_graph, name='citation_graph'),
path('papers/saved/', views.api_saved_papers_list, name='saved_papers'),
]