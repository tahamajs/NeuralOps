from django.urls import path
from . import views

urlpatterns = [
    # Core CRUD
    path('v1/notes/', views.versioned_view(views.note_list)),
    path('v1/notes/<uuid:note_id>/', views.versioned_view(views.note_detail)),
    path('v1/notes/<uuid:note_id>/react/', views.versioned_view(views.note_reaction)),
    path('v1/notes/<uuid:note_id>/upload-image/', views.versioned_view(views.upload_note_image)),
    path('v1/notes/<uuid:note_id>/favorite/', views.versioned_view(views.toggle_favorite)),
    path('v1/notes/<uuid:note_id>/comments/', views.versioned_view(views.note_comments)),
    path('v1/notes/runs/<uuid:run_id>/', views.versioned_view(views.run_notes)),

    # Version history
    path('v1/notes/<uuid:note_id>/versions/', views.versioned_view(views.note_versions)),
    path('v1/notes/<uuid:note_id>/versions/<int:version_number>/', views.versioned_view(views.note_version_detail)),

    # Mentions
    path('v1/notes/<uuid:note_id>/mentions/', views.versioned_view(views.note_mentions)),
    path('v1/mentions/<uuid:mention_id>/read/', views.versioned_view(views.mark_mention_read)),

    # Sharing
    path('v1/notes/<uuid:note_id>/share/', views.versioned_view(views.share_note)),
    path('v1/notes/<uuid:note_id>/unshare/', views.versioned_view(views.unshare_note)),
    path('v1/notes/shared-with-me/', views.versioned_view(views.shared_with_me)),

    # Replies
    path('v1/notes/<uuid:note_id>/replies/', views.versioned_view(views.note_replies)),

    # Restore
    path('v1/notes/<uuid:note_id>/restore/', views.versioned_view(views.restore_note)),

    # Export
    path('v1/notes/<uuid:note_id>/export/pdf/', views.versioned_view(views.export_note_pdf)),
    path('v1/notes/<uuid:note_id>/export/md/', views.versioned_view(views.export_note_markdown)),

    # Auto-Generate Reports
    path('v1/notes/generate-report/<uuid:run_id>/', views.versioned_view(views.generate_run_report)),

    # Templates
    path('v1/templates/', views.versioned_view(views.note_templates)),
    path('v1/templates/<uuid:template_id>/render/', views.versioned_view(views.render_template)),

    # Favorites
    path('v1/notes/favorites/', views.versioned_view(views.favorite_notes)),

    # Full-Text Search
    path('v1/search/', views.versioned_view(views.fulltext_search)),

    # Notifications
    path('v1/notifications/', views.versioned_view(views.my_notifications)),
    path('v1/notifications/read-all/', views.versioned_view(views.mark_all_notifications_read)),

    # Dashboard
    path('v1/dashboard/', views.versioned_view(views.notes_dashboard)),

    # Bulk operations
    path('v1/notes/bulk/', views.versioned_view(views.bulk_notes)),
    path('v1/notes/bulk/tags/', views.versioned_view(views.bulk_tags)),

    # Statistics
    path('v1/notes/statistics/', views.versioned_view(views.notes_statistics)),
    path('v1/notes/recent/', views.versioned_view(views.recent_notes)),

    # Webhooks
    path('v1/webhooks/', views.versioned_view(views.webhook_list)),
    path('v1/webhooks/<uuid:webhook_id>/', views.versioned_view(views.webhook_delete)),
    path('v1/webhooks/<uuid:webhook_id>/test/', views.versioned_view(views.test_webhook)),



    # Notes/urls.py – add these patterns

    # HuggingFace upload
    path('v1/notes/<uuid:note_id>/upload-hf/', views.versioned_view(views.upload_note_to_huggingface)),

    # Workspaces
    path('v1/workspaces/', views.versioned_view(views.workspace_list)),
    path('v1/workspaces/<uuid:workspace_id>/add-note/<uuid:note_id>/', views.versioned_view(views.workspace_add_note)),

    # Tasks
    path('v1/notes/<uuid:note_id>/tasks/', views.versioned_view(views.task_list)),
    path('v1/notes/<uuid:note_id>/tasks/<uuid:task_id>/', views.versioned_view(views.task_update)),

    # Model Registry
    path('v1/models/', views.versioned_view(views.model_registry_list)),

    # Note Links
    path('v1/notes/links/', views.versioned_view(views.create_note_link)),
]