# reviews/serializers.py
from rest_framework import serializers
from .models import *


class ReviewCommentSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    class Meta:
        model = ReviewComment
        fields = ['id', 'review', 'author', 'author_username', 'content',
                  'is_resolved', 'created_at']
        read_only_fields = ['id', 'author', 'created_at']


class ReviewSerializer(serializers.ModelSerializer):
    reviewer_username = serializers.CharField(source='reviewer.username', read_only=True)
    helpful_up = serializers.SerializerMethodField()
    helpful_down = serializers.SerializerMethodField()
    comments = ReviewCommentSerializer(many=True, read_only=True)

    class Meta:
        model = Review
        fields = ['id', 'review_request', 'reviewer', 'reviewer_username',
                  'overall_score', 'originality_score', 'soundness_score',
                  'presentation_score', 'significance_score', 'reproducibility_score',
                  'summary', 'strengths', 'weaknesses', 'suggestions', 'questions',
                  'is_anonymous', 'status', 'version', 'helpful_up', 'helpful_down',
                  'comments', 'created_at', 'updated_at']
        read_only_fields = ['id', 'reviewer', 'review_request', 'status', 'version',
                            'created_at', 'updated_at']

    def get_helpful_up(self, obj):
        return obj.helpfulness_votes.filter(direction='up').count()

    def get_helpful_down(self, obj):
        return obj.helpfulness_votes.filter(direction='down').count()


class ReviewRequestListSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    completed_reviews_count = serializers.SerializerMethodField()
    average_score = serializers.FloatField(read_only=True)

    class Meta:
        model = ReviewRequest
        fields = ['id', 'title', 'author', 'author_username', 'submission_type',
                  'tags', 'status', 'visibility', 'average_score',
                  'completed_reviews_count', 'view_count', 'comment_count',
                  'created_at', 'updated_at']

    def get_completed_reviews_count(self, obj):
        return obj.completed_reviews.count()


class ReviewRequestDetailSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    reviews = serializers.SerializerMethodField()
    average_score = serializers.FloatField(read_only=True)

    class Meta:
        model = ReviewRequest
        fields = ['id', 'author', 'author_username', 'title', 'abstract',
                  'submission_type', 'paper_url', 'file', 'tags', 'metadata',
                  'status', 'visibility', 'reviewers', 'reviewers_required',
                  'review_deadline', 'average_score', 'reviews',
                  'view_count', 'comment_count', 'created_at', 'updated_at']
        read_only_fields = ['id', 'author', 'status', 'average_score', 'view_count',
                            'comment_count', 'created_at', 'updated_at']

    def get_reviews(self, obj):
        reviews = obj.reviews.filter(status='completed')
        return ReviewSerializer(reviews, many=True).data


class ReviewTemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewTemplate
        fields = '__all__'
        read_only_fields = ['id', 'created_by', 'usage_count', 'created_at']


class ReviewBadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewBadge
        fields = '__all__'


class ReviewerBadgeSerializer(serializers.ModelSerializer):
    badge_detail = ReviewBadgeSerializer(source='badge', read_only=True)
    class Meta:
        model = ReviewerBadge
        fields = ['id', 'reviewer', 'badge', 'badge_detail', 'awarded_at']
        read_only_fields = ['id', 'awarded_at']