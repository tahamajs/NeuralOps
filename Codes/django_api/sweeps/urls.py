from django.urls import path
from . import views

urlpatterns = [
    path('sweeps/', views.sweep_list, name='sweep_list'),
    path('sweeps/<uuid:sweep_id>/', views.sweep_detail, name='sweep_detail'),
    path('sweeps/<uuid:sweep_id>/trials/', views.sweep_trials, name='sweep_trials'),
    path('sweeps/<uuid:sweep_id>/importance/', views.sweep_importance, name='sweep_importance'),
]