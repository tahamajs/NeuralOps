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
]