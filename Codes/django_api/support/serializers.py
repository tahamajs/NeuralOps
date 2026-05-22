from rest_framework import serializers
from .models import SupportTicket, TicketMessage, Solution, Feedback


class TicketMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TicketMessage
        fields = ['id', 'ticket', 'role', 'content', 'metadata', 'created_at']
        read_only_fields = ['id', 'ticket', 'role', 'metadata', 'created_at']


class SupportTicketSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    messages = TicketMessageSerializer(many=True, read_only=True)

    class Meta:
        model = SupportTicket
        fields = [
            'id', 'user', 'username', 'title', 'description', 'code_snippet',
            'log_attachment', 'tags', 'status', 'ai_summary', 'ai_steps',
            'web_sources', 'messages', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'user', 'username', 'status', 'ai_summary',
                            'ai_steps', 'web_sources', 'messages', 'created_at', 'updated_at']


class SupportTicketCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SupportTicket
        fields = ['title', 'description', 'code_snippet', 'log_attachment', 'tags']


class SolutionSerializer(serializers.ModelSerializer):
    ticket_title = serializers.CharField(source='ticket.title', read_only=True)

    class Meta:
        model = Solution
        fields = ['id', 'ticket', 'ticket_title', 'content', 'upvotes', 'created_at']
        read_only_fields = ['id', 'ticket', 'ticket_title', 'upvotes', 'created_at']


class FeedbackSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Feedback
        fields = ['id', 'ticket', 'user', 'username', 'rating', 'comment', 'created_at']
        read_only_fields = ['id', 'user', 'username', 'created_at']