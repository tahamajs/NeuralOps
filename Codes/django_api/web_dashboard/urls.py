from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    # ---------------------- HTML Pages ----------------------
    path('', views.index, name='dashboard_index'),
    path('create/', views.create_job, name='dashboard_create_job'),

    # Job / Run detail pages
    path('dashboard/job/<uuid:job_id>/', views.job_detail, name='dashboard_job_detail'),
    path('dashboard/run/<uuid:job_id>/', views.job_detail, name='dashboard_job_detail_run'),

    # Bootstrap dashboard
    path('dashboard/', views.dashboard, name='dashboard_dashboard'),

    # Statistics
    path('statistics/', views.statistics, name='dashboard_statistics'),

    # Auth pages
    path('login/', views.login_page, name='dashboard_login'),
    path('register/', views.register_page, name='dashboard_register'),

    # ---------------------- REST API Endpoints ----------------------
    path('api/jobs/', views.dashboard_api_jobs, name='dashboard_api_jobs'),
    path('api/register/', views.api_register, name='dashboard_api_register'),
    path('api/login/', views.api_login, name='dashboard_api_login'),
    path('api/logout/', views.api_logout, name='dashboard_api_logout'),

    # Dynamic config for launch page
    path('api/config/all/', views.launch_config_options, name='launch_config_options'),

    # Dashboard widgets
    path('api/widgets/', views.get_dashboard_widgets, name='dashboard_widgets'),
    path('api/widgets/layout/', views.save_dashboard_layout, name='dashboard_widgets_layout'),
    path('api/widgets/<uuid:widget_id>/data/', views.widget_data, name='widget_data'),

    # Optional extra endpoints (used by some frontend components)
    path('api/widgets/<uuid:widget_id>/refresh/', views.refresh_widget, name='refresh_widget'),
    path('api/widgets/<uuid:widget_id>/config/', views.widget_config, name='widget_config'),
    path('api/dashboard/stats/', views.dashboard_stats, name='dashboard_stats'),
    path('api/dashboard/settings/', views.get_dashboard_settings, name='dashboard_settings'),
    path('api/dashboard/settings/save/', views.save_dashboard_settings, name='save_dashboard_settings'),
]

# Serve static/media in development only
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)