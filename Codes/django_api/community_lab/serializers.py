from rest_framework import serializers
from .models import *


class AgentRoleSerializer(serializers.ModelSerializer):
    action_count = serializers.SerializerMethodField()

    class Meta:
        model = AgentRole
        fields = [
            'id', 'lab', 'name', 'role_type', 'description',
            'system_prompt', 'model_name', 'temperature',
            'tools_allowed', 'is_active', 'action_count', 'created_at',
        ]
        read_only_fields = ['id', 'lab', 'created_at']

    def get_action_count(self, obj):
        return obj.actions.count()


class ResearchLabListSerializer(serializers.ModelSerializer):
    creator_username = serializers.CharField(source='creator.username', read_only=True)
    member_count = serializers.IntegerField(read_only=True)
    active_tasks = serializers.SerializerMethodField()

    class Meta:
        model = ResearchLab
        fields = [
            'id', 'name', 'description', 'creator', 'creator_username',
            'status', 'visibility', 'member_count', 'active_tasks',
            'objective', 'max_cycles', 'budget_limit_cents', 'spent_cents',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'creator', 'created_at', 'updated_at']

    def get_active_tasks(self, obj):
        return obj.agent_actions.filter(action_type='experiment').count()


class ResearchLabDetailSerializer(serializers.ModelSerializer):
    creator_username = serializers.CharField(source='creator.username', read_only=True)
    agent_roles = AgentRoleSerializer(many=True, read_only=True)
    human_collaborator_usernames = serializers.SerializerMethodField()

    class Meta:
        model = ResearchLab
        fields = [
            'id', 'name', 'description', 'creator', 'creator_username',
            'prompt', 'objective', 'status', 'visibility',
            'human_collaborators', 'human_collaborator_usernames',
            'agent_roles', 'max_cycles', 'budget_limit_cents', 'spent_cents',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'creator', 'created_at', 'updated_at']

    def get_human_collaborator_usernames(self, obj):
        return [u.username for u in obj.human_collaborators.all()]


class AgentActionSerializer(serializers.ModelSerializer):
    agent_name = serializers.CharField(source='agent_role.name', read_only=True)

    class Meta:
        model = AgentAction
        fields = [
            'id', 'lab', 'agent_role', 'agent_name', 'action_type',
            'input_data', 'output_summary', 'tokens_used', 'cost_cents',
            'success', 'cycle_number', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class ChatMessageSerializer(serializers.ModelSerializer):
    sender_name = serializers.SerializerMethodField()

    class Meta:
        model = ChatMessage
        fields = [
            'id', 'room', 'sender_type', 'sender_user', 'sender_agent',
            'sender_name', 'content', 'metadata', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']

    def get_sender_name(self, obj):
        if obj.sender_type == 'human' and obj.sender_user:
            return obj.sender_user.username
        elif obj.sender_type == 'agent' and obj.sender_agent:
            return obj.sender_agent.name
        return 'System'


class ChatRoomSerializer(serializers.ModelSerializer):
    last_message = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatRoom
        fields = [
            'id', 'lab', 'name', 'description', 'is_private',
            'last_message', 'unread_count', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_last_message(self, obj):
        last = obj.messages.order_by('-created_at').first()
        if last:
            return ChatMessageSerializer(last).data
        return None

    def get_unread_count(self, obj):
        request = self.context.get('request')
        if request:
            # Simplified: count all messages as unread for now
            return obj.messages.count()
        return 0


class WorkflowGraphSerializer(serializers.ModelSerializer):
    creator_username = serializers.CharField(source='creator.username', read_only=True)

    class Meta:
        model = WorkflowGraph
        fields = [
            'id', 'lab', 'name', 'description', 'creator', 'creator_username',
            'nodes', 'edges', 'is_template', 'usage_count',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'creator', 'usage_count', 'created_at', 'updated_at']


class DiscoveryReviewSerializer(serializers.ModelSerializer):
    reviewer_username = serializers.CharField(source='reviewer.username', read_only=True)

    class Meta:
        model = DiscoveryReview
        fields = [
            'id', 'submission', 'reviewer', 'reviewer_username',
            'overall_score', 'methodology_score', 'reproducibility_score',
            'comments', 'is_public', 'created_at',
        ]
        read_only_fields = ['id', 'reviewer', 'created_at']


class DiscoverySubmissionSerializer(serializers.ModelSerializer):
    submitted_by_username = serializers.CharField(source='submitted_by.username', read_only=True)
    lab_name = serializers.CharField(source='lab.name', read_only=True)
    reviews = DiscoveryReviewSerializer(many=True, read_only=True)
    average_score = serializers.SerializerMethodField()

    class Meta:
        model = DiscoverySubmission
        fields = [
            'id', 'lab', 'lab_name', 'title', 'abstract', 'methodology',
            'results', 'reproducibility_score', 'submitted_by', 'submitted_by_username',
            'status', 'upvotes', 'downvotes', 'comment_count',
            'reviews', 'average_score', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'submitted_by', 'upvotes', 'downvotes',
                            'comment_count', 'created_at', 'updated_at']

    def get_average_score(self, obj):
        reviews = obj.reviews.all()
        if not reviews:
            return None
        return round(reviews.aggregate(a=models.Avg('overall_score'))['a'], 1)


class LabInvitationSerializer(serializers.ModelSerializer):
    invited_by_username = serializers.CharField(source='invited_by.username', read_only=True)

    class Meta:
        model = LabInvitation
        fields = [
            'id', 'lab', 'email', 'invited_by', 'invited_by_username',
            'status', 'expires_at', 'created_at',
        ]
        read_only_fields = ['id', 'invited_by', 'status', 'created_at']