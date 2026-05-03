from django.urls import path
from ..django_api import views

urlpatterns = [
    path('', views.index, name='index'),
    path('job/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('create/', views.create_job, name='create_job'),
    path('api/jobs/', views.dashboard_api_jobs, name='api_jobs'),
]