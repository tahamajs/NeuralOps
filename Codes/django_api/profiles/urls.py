# profiles/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('me/', views.my_profile, name='my_profile'),
    path('me/avatar/', views.upload_avatar, name='upload_avatar'),
    path('me/avatar/remove/', views.remove_avatar, name='remove_avatar'),
    path('search/', views.search_users, name='search_users'),
    path('user/<int:user_id>/', views.user_profile, name='user_profile'),
    path('username/<str:username>/', views.user_profile_by_username, name='user_profile_by_username'),
    path('social-login/', views.social_login_urls, name='social_login_urls'),
path('social-callback/', views.social_callback, name='social_callback'),

]