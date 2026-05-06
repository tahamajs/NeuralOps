from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    # Main dashboard – will be used by root include
    path('', views.index, name='dashboard_index'),

    # Create new experiment / run (wizard)
    path('create/', views.create_job, name='dashboard_create_job'),

    # Job (run) detail pages
    path('dashboard/job/<uuid:job_id>/', views.job_detail, name='dashboard_job_detail'),
    path('dashboard/run/<uuid:job_id>/', views.job_detail, name='dashboard_job_detail_run'),

    # Full dashboard (bootstrap version)
    path('dashboard/', views.dashboard, name='dashboard_dashboard'),

    # Statistics page
    path('statistics/', views.statistics, name='dashboard_statistics'),

    # Authentication pages
    path('login/', views.login_page, name='dashboard_login'),
    path('register/', views.register_page, name='dashboard_register'),

    # API endpoints for the frontend (token‑based)
    path('api/jobs/', views.dashboard_api_jobs, name='dashboard_api_jobs'),
    path('api/register/', views.api_register, name='dashboard_api_register'),
    path('api/login/', views.api_login, name='dashboard_api_login'),
    path('api/logout/', views.api_logout, name='dashboard_api_logout'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)