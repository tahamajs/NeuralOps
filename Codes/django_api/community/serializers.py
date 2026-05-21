# community/serializers.py
from rest_framework import serializers
from .models import *


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name', 'slug', 'description', 'icon', 'is_official',
                  'question_count', 'parent', 'created_at']


class VoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vote
        fields = ['id', 'voter', 'question', 'answer', 'direction', 'created_at']
        read_only_fields = ['id', 'voter', 'created_at']


class CommentSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    class Meta:
        model = Comment
        fields = ['id', 'question', 'answer', 'author', 'author_username', 'body', 'created_at']
        read_only_fields = ['id', 'author', 'created_at']


class AnswerSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    author_reputation = serializers.SerializerMethodField()
    comment_count = serializers.SerializerMethodField()
    comments = CommentSerializer(many=True, read_only=True)

    class Meta:
        model = Answer
        fields = ['id', 'question', 'author', 'author_username', 'author_reputation',
                  'body', 'is_accepted', 'score', 'upvotes', 'downvotes',
                  'comment_count', 'comments', 'created_at', 'updated_at']
        read_only_fields = ['id', 'author', 'is_accepted', 'score', 'upvotes',
                            'downvotes', 'created_at', 'updated_at']

    def get_author_reputation(self, obj):
        return obj.author.reputation_events.aggregate(s=models.Sum('amount'))['s'] or 0

    def get_comment_count(self, obj):
        return obj.comments.count()


class QuestionListSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    tags_detail = TagSerializer(source='tags', many=True, read_only=True)
    has_accepted_answer = serializers.SerializerMethodField()

    class Meta:
        model = Question
        fields = ['id', 'title', 'author', 'author_username', 'tags_detail',
                  'status', 'score', 'view_count', 'answer_count',
                  'has_accepted_answer', 'has_bounty', 'created_at', 'updated_at',
                  'last_activity_at']

    def get_has_accepted_answer(self, obj):
        return obj.accepted_answer_id is not None


class QuestionDetailSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    author_reputation = serializers.SerializerMethodField()
    tags_detail = TagSerializer(source='tags', many=True, read_only=True)
    answers = serializers.SerializerMethodField()
    comments = CommentSerializer(many=True, read_only=True)
    bookmarked = serializers.SerializerMethodField()
    active_bounty = serializers.SerializerMethodField()

    class Meta:
        model = Question
        fields = ['id', 'title', 'body', 'author', 'author_username',
                  'author_reputation', 'tags_detail', 'status', 'close_reason',
                  'score', 'upvotes', 'downvotes', 'view_count', 'answer_count',
                  'accepted_answer', 'has_bounty', 'bookmarked', 'active_bounty',
                  'answers', 'comments', 'linked_paper', 'linked_run',
                  'created_at', 'updated_at', 'last_activity_at']
        read_only_fields = ['id', 'author', 'status', 'score', 'upvotes',
                            'downvotes', 'view_count', 'answer_count',
                            'created_at', 'updated_at', 'last_activity_at']

    def get_author_reputation(self, obj):
        return obj.author.reputation_events.aggregate(s=models.Sum('amount'))['s'] or 0

    def get_answers(self, obj):
        answers = obj.answers.all().order_by('-is_accepted', '-score')
        return AnswerSerializer(answers, many=True).data

    def get_bookmarked(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.bookmarks.filter(user=request.user).exists()
        return False

    def get_active_bounty(self, obj):
        bounty = obj.bounties.filter(awarded_to__isnull=True,
                                       expires_at__gt=timezone.now()).first()
        if bounty:
            return {'amount': bounty.amount, 'offerer': bounty.offerer.username,
                    'reason': bounty.reason, 'expires_at': bounty.expires_at}
        return None


class BountySerializer(serializers.ModelSerializer):
    offerer_username = serializers.CharField(source='offerer.username', read_only=True)

    class Meta:
        model = Bounty
        fields = ['id', 'question', 'offerer', 'offerer_username', 'amount',
                  'reason', 'notice', 'awarded_to', 'expires_at', 'created_at']
        read_only_fields = ['id', 'offerer', 'awarded_to', 'created_at']


class BadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Badge
        fields = ['id', 'name', 'description', 'tier', 'category', 'icon',
                  'tag', 'condition', 'created_at']


class UserBadgeSerializer(serializers.ModelSerializer):
    badge_detail = BadgeSerializer(source='badge', read_only=True)
    class Meta:
        model = UserBadge
        fields = ['id', 'user', 'badge', 'badge_detail', 'question', 'answer',
                  'awarded_at', 'count']


class ReputationEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReputationEvent
        fields = ['id', 'user', 'event_type', 'amount', 'question', 'answer', 'created_at']


class TrustLevelSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    class Meta:
        model = TrustLevel
        fields = ['user', 'username', 'level', 'updated_at']