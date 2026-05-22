from django.urls import path
from . import views

urlpatterns = [
    # Pipelines
    path('pipelines/', views.pipeline_list, name='pipeline_list'),
    path('pipelines/<uuid:pipeline_id>/', views.pipeline_detail, name='pipeline_detail'),
    path('pipelines/<uuid:pipeline_id>/launch/', views.launch_pipeline, name='launch_pipeline'),
    path('pipelines/<uuid:pipeline_id>/pause/', views.pause_pipeline, name='pause_pipeline'),
    path('pipelines/<uuid:pipeline_id>/resume/', views.resume_pipeline, name='resume_pipeline'),

    # Steps
    path('pipelines/<uuid:pipeline_id>/steps/', views.step_list, name='step_list'),
    path('steps/<uuid:step_id>/update/', views.update_step, name='update_step'),

    # Artifacts
    path('pipelines/<uuid:pipeline_id>/artifacts/', views.artifact_list, name='artifact_list'),
    path('artifacts/<uuid:artifact_id>/', views.artifact_detail, name='artifact_detail'),

    # Templates
    path('templates/', views.template_list, name='pipeline_template_list'),
]