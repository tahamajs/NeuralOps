from rest_framework import serializers
from .models import *


class LLMProviderSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLMProvider
        fields = '__all__'


class LLMModelSerializer(serializers.ModelSerializer):
    provider_name = serializers.CharField(source='provider.name', read_only=True)

    class Meta:
        model = LLMModel
        fields = '__all__'


class PromptVersionSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)

    class Meta:
        model = PromptVersion
        fields = '__all__'


class PromptTemplateSerializer(serializers.ModelSerializer):
    versions = PromptVersionSerializer(many=True, read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = PromptTemplate
        fields = '__all__'


class PlaygroundRunSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source='model.display_name', read_only=True)

    class Meta:
        model = PlaygroundRun
        fields = '__all__'


class PlaygroundSessionSerializer(serializers.ModelSerializer):
    runs = PlaygroundRunSerializer(many=True, read_only=True)

    class Meta:
        model = PlaygroundSession
        fields = '__all__'


class ToolDefinitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ToolDefinition
        fields = '__all__'


class StructuredOutputSchemaSerializer(serializers.ModelSerializer):
    class Meta:
        model = StructuredOutputSchema
        fields = '__all__'


class EvalTestCaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvalTestCase
        fields = '__all__'


class EvalSuiteSerializer(serializers.ModelSerializer):
    test_cases = EvalTestCaseSerializer(many=True, read_only=True)
    test_count = serializers.SerializerMethodField()

    class Meta:
        model = EvalSuite
        fields = '__all__'

    def get_test_count(self, obj):
        return obj.test_cases.count()


class EvalResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvalResult
        fields = '__all__'


class EvalRunSerializer(serializers.ModelSerializer):
    results = EvalResultSerializer(many=True, read_only=True)
    model_name = serializers.CharField(source='model.display_name', read_only=True)
    suite_name = serializers.CharField(source='suite.name', read_only=True)

    class Meta:
        model = EvalRun
        fields = '__all__'


class MemoryEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = MemoryEntry
        fields = '__all__'


class UsageRecordSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source='model.display_name', read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = UsageRecord
        fields = '__all__'