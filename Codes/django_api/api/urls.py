from django.urls import include, path
from . import views

urlpatterns = [
    path("auth/register/", views.register),
    path("auth/login/", views.login),
    path("auth/logout/", views.logout),
    path("auth/user/", views.user_info),
    path("auth/profile/", views.update_profile),
    path("auth/change-password/", views.change_password),
    path("auth/refresh-token/", views.refresh_token),

    path("config/attentions/", views.list_attentions),
    path("config/compressors/", views.list_compressors),
    path("config/models/", views.list_models),
    path("config/tasks/", views.list_tasks),
    path("config/methods/", views.list_methods),
    path("config/presets/", views.list_presets),

    # System routes are now in jobscheduler.urls
    path("health/", views.health_check),
    path("version/", views.version),

    # Jobscheduler – before experiment.urls to avoid shadowing
    path('', include('jobscheduler.urls')),

    # Experiment leftover (only unique endpoints)
    path('', include('experiment.urls')),
]