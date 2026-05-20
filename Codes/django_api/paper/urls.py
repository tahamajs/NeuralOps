from django.urls import path

from . import views


urlpatterns = [
    path('projects/', views.project_list, name='paper_project_list'),
    path('projects/<uuid:project_id>/', views.project_detail, name='paper_project_detail'),
    path('projects/<uuid:project_id>/files/', views.file_list, name='paper_file_list'),
    path('projects/<uuid:project_id>/files/<path:filename>/', views.file_detail, name='paper_file_detail'),
    path('projects/<uuid:project_id>/compile/', views.compile_project, name='paper_compile'),
    path('projects/<uuid:project_id>/output/pdf/', views.output_pdf, name='paper_output_pdf'),
    path('projects/<uuid:project_id>/output/log/', views.output_log, name='paper_output_log'),
    path('projects/<uuid:project_id>/export/', views.export_project, name='paper_export'),
    path('projects/<uuid:project_id>/collaborators/add/', views.add_collaborator, name='paper_add_collaborator'),
    path('projects/<uuid:project_id>/collaborators/remove/', views.remove_collaborator, name='paper_remove_collaborator'),
    path('projects/<uuid:project_id>/compiles/', views.compile_history, name='paper_compile_history'),
    path('templates/', views.template_list, name='paper_template_list'),
    path('templates/<uuid:template_id>/use/', views.use_template, name='paper_use_template'),
]
