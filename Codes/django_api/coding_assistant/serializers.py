from rest_framework import serializers
from .models import CodeGenerationSession, CodeMessage, GeneratedCode, CodeReview, TestGeneration


class CodeMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = CodeMessage
        fields = ['id', 'session', 'role', 'content', 'language', 'tokens_used', 'created_at']
        read_only_fields = ['id', 'tokens_used', 'created_at']


class GeneratedCodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedCode
        fields = ['id', 'session', 'filename', 'content', 'language', 'description', 'status', 'version', 'created_at']
        read_only_fields = ['id', 'version', 'created_at', 'updated_at']


class CodeReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = CodeReview
        fields = ['id', 'generated_code', 'issues', 'suggestions', 'overall_assessment', 'created_at']
        read_only_fields = ['id', 'created_at']


class TestGenerationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestGeneration
        fields = ['id', 'generated_code', 'test_content', 'test_framework', 'coverage_estimate', 'created_at']
        read_only_fields = ['id', 'created_at']


class CodeGenerationSessionSerializer(serializers.ModelSerializer):
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = CodeGenerationSession
        fields = ['id', 'title', 'context', 'message_count', 'last_message', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last = obj.messages.order_by('-created_at').first()
        if last:
            return {'role': last.role, 'content': last.content[:100]}
        return None


class CodeGenerationSessionDetailSerializer(serializers.ModelSerializer):
    messages = CodeMessageSerializer(many=True, read_only=True)
    generated_files = GeneratedCodeSerializer(many=True, read_only=True)

    class Meta:
        model = CodeGenerationSession
        fields = ['id', 'title', 'context', 'messages', 'generated_files', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']