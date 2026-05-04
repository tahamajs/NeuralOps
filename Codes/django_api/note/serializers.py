# notes/serializers.py
from rest_framework import serializers
from .models import ResearcherNote, NoteImage, WebhookSubscription

class ResearcherNoteSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    run_id = serializers.UUIDField(source='run.id', read_only=True)
    
    class Meta:
        model = ResearcherNote
        fields = '__all__'
        read_only_fields = ('id', 'author', 'created_at', 'updated_at', 'slug', 'version')

class NoteImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = NoteImage
        fields = '__all__'
        read_only_fields = ('id', 'uploaded_at')

class WebhookSubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebhookSubscription
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'last_triggered_at')