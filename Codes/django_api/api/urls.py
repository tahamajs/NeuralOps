from django.urls import include, path
from . import views

urlpatterns = [
    # Authentication
    path("auth/register/", views.register),
    path("auth/login/", views.login),
    path("auth/logout/", views.logout),
    path("auth/user/", views.user_info),
    path("auth/profile/", views.update_profile),
    path("auth/change-password/", views.change_password),
    path("auth/refresh-token/", views.refresh_token),
    path("config/trainers/", views.list_trainers),
    path("config/methods/", views.list_methods),

    # Configuration
    path("config/attentions/", views.list_attentions),
    path("config/compressors/", views.list_compressors),
    path("config/models/", views.list_models),
    path("config/tasks/", views.list_tasks),
    path("config/methods/", views.list_methods),
    path("config/presets/", views.list_presets),

    # Health / Version
    path("health/", views.health_check),
    path("version/", views.version),

    # Jobscheduler – explicit paths to avoid conflicts
    path('', include('jobscheduler.urls')),

    # Experiment leftovers (runs, preview, etc.)
    path('', include('experiment.urls')),
]