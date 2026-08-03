from django.contrib import admin

from .models import (
    Artifact,
    NoteComment,
    NoteLink,
    Notification,
    Protocol,
    ShareableLink,
)


@admin.register(ShareableLink)
class ShareableLinkAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "note",
        "token_preview",
        "is_active",
        "view_count",
        "created_at",
    )
    list_filter = ("is_active",)
    search_fields = ("token", "note__title")

    def token_preview(self, obj):
        return obj.token[:20] + "..."


@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "file_type", "file_size_mb", "note", "created_at")


@admin.register(Protocol)
class ProtocolAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "steps_count",
        "is_public",
        "created_by",
        "created_at",
    )

    def steps_count(self, obj):
        return len(obj.steps)


@admin.register(NoteLink)
class NoteLinkAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "target", "link_type", "created_at")


@admin.register(NoteComment)
class NoteCommentAdmin(admin.ModelAdmin):
    list_display = ("id", "note", "author", "content_preview", "created_at")

    def content_preview(self, obj):
        return obj.content[:60]


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ("id", "recipient", "type", "is_read", "created_at")
    list_filter = ("type", "is_read")
