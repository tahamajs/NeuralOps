from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    # Main dashboard (list of runs / experiments)
    path('', views.index, name='index'),

    # Create new experiment / run (wizard)
    path('create/', views.create_job, name='create_job'),

    # Job (run) detail pages – both /dashboard/job/ and /dashboard/run/ work
    path('dashboard/job/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('dashboard/run/<uuid:job_id>/', views.job_detail, name='job_detail_run'),

    # Alternative full dashboard (bootstrap version)
    path('dashboard/', views.dashboard, name='dashboard'),

    # Statistics page (aggregated metrics)
    path('statistics/', views.statistics, name='statistics'),

    # Authentication pages
    path('login/', views.login_page, name='login'),
    path('register/', views.register_page, name='register'),

    # API endpoints for the frontend (token‑based)
    path('api/jobs/', views.dashboard_api_jobs, name='api_jobs'),
    path('api/register/', views.api_register, name='api_register'),
    path('api/login/', views.api_login, name='api_login'),
    path('api/logout/', views.api_logout, name='api_logout'),
]

# Serve static files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)