from rest_framework import serializers
from .models import ResearcherNote

class ResearcherNoteSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    run_id = serializers.UUIDField(source='run.id', read_only=True)

    class Meta:
        model = ResearcherNote
        fields = '__all__'
        read_only_fields = ('id', 'author', 'created_at', 'updated_at')