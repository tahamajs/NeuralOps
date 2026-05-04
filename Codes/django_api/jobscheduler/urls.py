from django.urls import path
from . import views

urlpatterns = [
    path('jobs/', views.job_list),
    path('jobs/<uuid:job_id>/', views.job_detail),
    path('jobs/<uuid:job_id>/priority/', views.job_detail),
    path('jobs/<uuid:job_id>/pause/', views.job_pause),
    path('jobs/<uuid:job_id>/resume/', views.job_resume),
    path('jobs/<uuid:job_id>/stop/', views.job_stop),
    path('jobs/<uuid:job_id>/retry/', views.job_retry),
    path('jobs/<uuid:job_id>/history/', views.job_history),
    path('jobs/<uuid:job_id>/notes/', views.job_notes),
    path('system/gpu/', views.gpu_info),
]