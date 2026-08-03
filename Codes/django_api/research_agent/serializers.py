from rest_framework import serializers

from .models import AgentMessage, AgentSession, GeneratedIdea


class AgentSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentSession
        fields = "__all__"


class AgentMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentMessage
        fields = "__all__"


class GeneratedIdeaSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedIdea
        fields = "__all__"
