from django.urls import path
from . import views

urlpatterns = [
    # Inference Endpoints
    path('endpoints/', views.endpoint_list, name='endpoint_list'),
    path('endpoints/<uuid:endpoint_id>/call/', views.call_endpoint, name='call_endpoint'),

    # Pipelines
    path('pipelines/<uuid:pipeline_id>/run/', views.run_pipeline, name='run_pipeline'),

    # Feature Store
    path('features/ingest/', views.feature_ingest, name='feature_ingest'),
    path('features/serve/', views.feature_serve, name='feature_serve'),

    # CT/CD
    path('ctcd/<uuid:pipeline_id>/trigger/', views.trigger_ctcd, name='trigger_ctcd'),

    # Distillation
    path('distillation/start/', views.start_distillation, name='start_distillation'),

    # Multi‑Node Training
    path('multi-node/start/', views.start_multi_node_training, name='start_multi_node_training'),
]