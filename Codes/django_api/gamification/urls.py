from django.urls import path
from . import views

urlpatterns = [
    # Badges
    path('badges/', views.badge_list, name='badge_list'),
    path('badges/mine/', views.my_badges, name='my_badges'),

    # Contributions
    path('contributions/', views.my_contributions, name='my_contributions'),
    path('points/', views.my_points_summary, name='my_points_summary'),

    # Streaks
    path('streak/', views.my_streak, name='my_streak'),

    # Leaderboard
    path('leaderboard/', views.leaderboard, name='leaderboard'),

    # Achievements & Premium Features
    path('unlocks/', views.my_unlocks, name='my_unlocks'),
    path('features/', views.premium_features, name='premium_features'),
    path('features/<str:feature_name>/access/', views.check_feature_access, name='check_feature_access'),
]