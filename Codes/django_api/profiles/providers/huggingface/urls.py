from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.oauth2_login, name='huggingface_login'),
    path('callback/', views.oauth2_callback, name='huggingface_callback'),
]