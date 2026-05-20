# paper/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Projects
    path('projects/', views.project_list, name='paper_project_list'),
    path('projects/<uuid:project_id>/', views.project_detail, name='paper_project_detail'),
    path('projects/<uuid:project_id>/clone/', views.clone_project, name='paper_clone_project'),

    # Files
    path('projects/<uuid:project_id>/files/', views.file_list, name='paper_file_list'),
    path('projects/<uuid:project_id>/files/<path:filename>/', views.file_detail, name='paper_file_detail'),
    path('projects/<uuid:project_id>/files/<path:filename>/rename/', views.rename_file, name='paper_rename_file'),

    # Compilation
    path('projects/<uuid:project_id>/compile/', views.compile_project, name='paper_compile'),
    path('projects/<uuid:project_id>/output/pdf/', views.output_pdf, name='paper_output_pdf'),
    path('projects/<uuid:project_id>/output/log/', views.output_log, name='paper_output_log'),
    path('projects/<uuid:project_id>/preview/pdf/', views.preview_pdf, name='paper_preview_pdf'),

    # Templates
    path('templates/', views.template_list, name='paper_template_list'),
    path('templates/<uuid:template_id>/use/', views.use_template, name='paper_use_template'),

    # Export
    path('projects/<uuid:project_id>/export/', views.export_project, name='paper_export'),

    # Word Count
    path('projects/<uuid:project_id>/wordcount/', views.word_count, name='paper_word_count'),

    # Collaboration
    path('projects/<uuid:project_id>/collaborators/add/', views.add_collaborator, name='paper_add_collaborator'),
    path('projects/<uuid:project_id>/collaborators/remove/', views.remove_collaborator, name='paper_remove_collaborator'),

    # Share Links (Public)
    path('projects/<uuid:project_id>/share-links/', views.create_share_link, name='paper_create_share_link'),
    path('projects/<uuid:project_id>/share-links/list/', views.list_share_links, name='paper_list_share_links'),
    path('projects/<uuid:project_id>/share-links/<uuid:link_id>/revoke/', views.revoke_share_link, name='paper_revoke_share_link'),
    # Public (no auth):
    path('shared/<str:token>/', views.view_shared_project, name='paper_view_shared'),
    path('shared/<str:token>/pdf/', views.shared_project_pdf, name='paper_shared_pdf'),

    # Git Sync
    path('projects/<uuid:project_id>/git/push/', views.git_push, name='paper_git_push'),
    path('projects/<uuid:project_id>/git/pull/', views.git_pull, name='paper_git_pull'),

    # Comments
    path('projects/<uuid:project_id>/comments/', views.comment_list, name='paper_comment_list'),
    path('projects/<uuid:project_id>/comments/<uuid:comment_id>/', views.comment_detail, name='paper_comment_detail'),

    # Snapshots (Version History)
    path('projects/<uuid:project_id>/snapshots/', views.snapshot_list, name='paper_snapshot_list'),
    path('projects/<uuid:project_id>/snapshots/<int:version_number>/', views.snapshot_detail, name='paper_snapshot_detail'),
    path('projects/<uuid:project_id>/snapshots/<int:version_number>/restore/', views.restore_snapshot, name='paper_restore_snapshot'),

    # Compile History
    path('projects/<uuid:project_id>/compiles/', views.compile_history, name='paper_compile_history'),

    # HuggingFace Upload
    path('projects/<uuid:project_id>/upload-hf/', views.upload_to_huggingface, name='paper_upload_hf'),
]