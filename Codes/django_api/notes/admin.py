from django.contrib import admin
from .models import ResearcherNote, NoteImage, WebhookSubscription

@admin.register(ResearcherNote)
class ResearcherNoteAdmin(admin.ModelAdmin):
    list_display = ('id', 'author', 'title', 'created_at', 'is_public')
    list_filter = ('is_public', 'pinned', 'author')
    search_fields = ('title', 'content')
    readonly_fields = ('slug', 'version')
    
@admin.register(NoteImage)
class NoteImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'note', 'uploaded_at')
    
@admin.register(WebhookSubscription)
class WebhookSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'url', 'is_active')