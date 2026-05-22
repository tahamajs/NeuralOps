from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Publication, Authorship, Citation, Review

User = get_user_model()


class AuthorshipSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)

    class Meta:
        model = Authorship
        fields = ['author', 'author_username', 'author_order', 'corresponding_author']


class PublicationSerializer(serializers.ModelSerializer):
    authors_detail = serializers.SerializerMethodField()
    citation_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Publication
        fields = ['id', 'title', 'abstract', 'publication_type', 'doi', 'arxiv_id',
                  'pdf_file', 'supplementary_files', 'repository_url', 'journal', 'conference',
                  'publication_date', 'citation_count', 'view_count', 'download_count',
                  'tags', 'authors_detail', 'created_at', 'updated_at']
        read_only_fields = ['id', 'citation_count', 'view_count', 'download_count', 'created_at', 'updated_at']

    def get_authors_detail(self, obj):
        return AuthorshipSerializer(obj.authorship_set.all(), many=True).data


class CitationSerializer(serializers.ModelSerializer):
    source_title = serializers.CharField(source='source.title', read_only=True)
    target_title = serializers.CharField(source='target.title', read_only=True)

    class Meta:
        model = Citation
        fields = ['id', 'source', 'target', 'source_title', 'target_title', 'created_at']


class ReviewSerializer(serializers.ModelSerializer):
    reviewer_username = serializers.CharField(source='reviewer.username', read_only=True)

    class Meta:
        model = Review
        fields = ['id', 'publication', 'reviewer', 'reviewer_username', 'content', 'rating', 'is_anonymous', 'created_at']
        read_only_fields = ['id', 'reviewer', 'created_at']