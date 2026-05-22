# literature_review/serializers.py
from rest_framework import serializers
from .models import *

class TopicSerializer(serializers.ModelSerializer):
    class Meta:
        model = Topic
        fields = '__all__'
        read_only_fields = ['id', 'created_at']

class ResearchNotebookSectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchNotebookSection
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']

class PaperReviewNoteSerializer(serializers.ModelSerializer):
    paper_title = serializers.CharField(source='paper.title', read_only=True)
    class Meta:
        model = PaperReviewNote
        fields = ['id', 'deep_review', 'paper', 'paper_title', 'notes', 'key_points', 'created_at']
        read_only_fields = ['id', 'created_at']

class DeepReviewSerializer(serializers.ModelSerializer):
    paper_notes = PaperReviewNoteSerializer(many=True, read_only=True)
    class Meta:
        model = DeepReview
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']

class LiteratureSearchSerializer(serializers.ModelSerializer):
    class Meta:
        model = LiteratureSearch
        fields = '__all__'
        read_only_fields = ['id', 'created_at']

class ResearchGapSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchGap
        fields = '__all__'
        read_only_fields = ['id', 'created_at']

class LiteratureProjectSerializer(serializers.ModelSerializer):
    topics = TopicSerializer(many=True, read_only=True)
    notebook_sections = ResearchNotebookSectionSerializer(many=True, read_only=True)
    deep_reviews = DeepReviewSerializer(many=True, read_only=True)
    searches = LiteratureSearchSerializer(many=True, read_only=True)
    gaps = ResearchGapSerializer(many=True, read_only=True)
    class Meta:
        model = LiteratureProject
        fields = '__all__'
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']