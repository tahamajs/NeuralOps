from rest_framework import serializers
from .models import FailureAnalysis, FixAction, ErrorPattern


class FixActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FixAction
        fields = [
            'id', 'analysis', 'action_type', 'description',
            'config_changes', 'is_applied', 'applied_at', 'applied_by', 'created_at',
        ]
        read_only_fields = ['id', 'is_applied', 'applied_at', 'applied_by', 'created_at']


class FailureAnalysisSerializer(serializers.ModelSerializer):
    fix_actions = FixActionSerializer(many=True, read_only=True)
    run_name = serializers.CharField(source='run.name', read_only=True)
    run_status = serializers.CharField(source='run.status', read_only=True)
    initiated_by_username = serializers.CharField(source='initiated_by.username', read_only=True)

    class Meta:
        model = FailureAnalysis
        fields = [
            'id', 'run', 'run_name', 'run_status', 'initiated_by',
            'initiated_by_username', 'status', 'root_cause_category',
            'root_cause_detail', 'confidence', 'error_logs',
            'anomaly_patterns', 'suggestions', 'fix_actions', 'created_at',
        ]
        read_only_fields = ['id', 'status', 'root_cause_category',
                            'root_cause_detail', 'confidence', 'error_logs',
                            'anomaly_patterns', 'suggestions', 'created_at']


class ErrorPatternSerializer(serializers.ModelSerializer):
    class Meta:
        model = ErrorPattern
        fields = ['id', 'pattern_name', 'regex_pattern', 'root_cause_category',
                  'description', 'suggestion', 'occurrences', 'is_active', 'created_at']
        read_only_fields = ['id', 'occurrences', 'created_at']