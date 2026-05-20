# messaging/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('chats/', views.chat_list, name='chat_list'),
    path('chats/start/', views.start_chat, name='start_chat'),
    path('chats/<uuid:room_id>/', views.chat_detail, name='chat_detail'),
    path('chats/<uuid:room_id>/send/', views.send_message, name='send_message'),
    path('chats/<uuid:room_id>/read/', views.mark_read, name='mark_read'),
    path('chats/<uuid:room_id>/delete/', views.delete_chat, name='delete_chat'),
]