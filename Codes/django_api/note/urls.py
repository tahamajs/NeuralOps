# notes/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # ------------------------------------------------------------------
    # Core note endpoints
    # ------------------------------------------------------------------
    path('v1/notes/', views.versioned_view(views.note_list)),
    path('v1/notes/<uuid:note_id>/', views.versioned_view(views.note_detail)),
    path('v1/notes/<uuid:note_id>/react/', views.versioned_view(views.note_reaction)),
    path('v1/notes/<uuid:note_id>/upload-image/', views.versioned_view(views.upload_note_image)),

    # ------------------------------------------------------------------
    # Note versions (history)
    # ------------------------------------------------------------------
    path('v1/notes/<uuid:note_id>/versions/', views.versioned_view(views.note_versions)),
    path('v1/notes/<uuid:note_id>/versions/<int:version_number>/', views.versioned_view(views.note_version_detail)),

    # ------------------------------------------------------------------
    # Mentions
    # ------------------------------------------------------------------
    path('v1/notes/<uuid:note_id>/mentions/', views.versioned_view(views.note_mentions)),
    path('v1/mentions/<uuid:mention_id>/read/', views.versioned_view(views.mark_mention_read)),

    # ------------------------------------------------------------------
    # Replies (threading)
    # ------------------------------------------------------------------
    path('v1/notes/<uuid:note_id>/replies/', views.versioned_view(views.note_replies)),

    # ------------------------------------------------------------------
    # Restore soft-deleted note
    # ------------------------------------------------------------------
    path('v1/notes/<uuid:note_id>/restore/', views.versioned_view(views.restore_note)),

    # ------------------------------------------------------------------
    # Export note as PDF
    # ------------------------------------------------------------------
    path('v1/notes/<uuid:note_id>/export/pdf/', views.versioned_view(views.export_note_pdf)),

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    path('v1/notes/bulk/', views.versioned_view(views.bulk_notes)),
    path('v1/notes/bulk/tags/', views.versioned_view(views.bulk_tags)),

    # ------------------------------------------------------------------
    # Statistics & recent notes
    # ------------------------------------------------------------------
    path('v1/notes/statistics/', views.versioned_view(views.notes_statistics)),
    path('v1/notes/recent/', views.versioned_view(views.recent_notes)),

    # ------------------------------------------------------------------
    # Webhook management
    # ------------------------------------------------------------------
    path('v1/webhooks/', views.versioned_view(views.webhook_list)),
    path('v1/webhooks/<uuid:webhook_id>/', views.versioned_view(views.webhook_delete)),
    path('v1/webhooks/<uuid:webhook_id>/test/', views.versioned_view(views.test_webhook)),
]