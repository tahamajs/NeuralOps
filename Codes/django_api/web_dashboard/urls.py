from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('create/', views.create_job, name='create_job'),
    path('dashboard/job/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('statistics/', views.statistics, name='statistics'),
    path('login/', views.login_page, name='login'),
    path('register/', views.register_page, name='register'),
    path('api/jobs/', views.dashboard_api_jobs, name='api_jobs'),
    path('api/register/', views.api_register, name='api_register'),
    path('api/login/', views.api_login, name='api_login'),
    path('api/logout/', views.api_logout, name='api_logout'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)