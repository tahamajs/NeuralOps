from django.urls import path
from . import views

urlpatterns = [
    # Anomaly Detection
    path('anomalies/', views.anomaly_list, name='anomaly_list'),
    path('anomalies/detect/<uuid:run_id>/', views.trigger_anomaly_detection, name='anomaly_detect'),
    path('anomalies/summary/', views.anomaly_summary, name='anomaly_summary'),

    # HPO Runs
    path('hpo/', views.hpo_run_list, name='hpo_run_list'),
    path('hpo/<uuid:hpo_run_id>/', views.hpo_run_detail, name='hpo_run_detail'),
    path('hpo/<uuid:hpo_run_id>/best/', views.hpo_best_result, name='hpo_best_result'),
    path('hpo/trials/<uuid:trial_id>/', views.hpo_trial_detail, name='hpo_trial_detail'),
    path('hpo/schema/', views.hpo_search_space_schema, name='hpo_schema'),
]