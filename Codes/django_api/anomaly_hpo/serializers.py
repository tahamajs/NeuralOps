from rest_framework import serializers
from .models import ExperimentLogAnalysis, HPORun, HPOTrial


class ExperimentLogAnalysisSerializer(serializers.ModelSerializer):
    run_name = serializers.CharField(source='run.name', read_only=True)

    class Meta:
        model = ExperimentLogAnalysis
        fields = [
            'id', 'run', 'run_name', 'anomaly_type', 'severity',
            'detected_at_step', 'description', 'suggested_fix',
            'auto_fix_applied', 'fix_successful', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class HPOTrialSerializer(serializers.ModelSerializer):
    class Meta:
        model = HPOTrial
        fields = [
            'id', 'hpo_run', 'trial_number', 'params',
            'objective_value', 'objective_metric', 'direction',
            'status', 'run', 'error_message',
            'started_at', 'completed_at', 'created_at',
        ]
        read_only_fields = ['id', 'status', 'objective_value',
                            'started_at', 'completed_at', 'created_at']


class HPORunSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    trial_count = serializers.SerializerMethodField()

    class Meta:
        model = HPORun
        fields = [
            'id', 'user', 'username', 'name', 'strategy', 'status',
            'experiment', 'search_space', 'objective_metric',
            'objective_direction', 'max_trials', 'base_config',
            'best_objective_value', 'best_trial', 'best_params',
            'trial_count', 'started_at', 'completed_at', 'created_at',
        ]
        read_only_fields = ['id', 'user', 'status', 'best_objective_value',
                            'best_trial', 'best_params',
                            'started_at', 'completed_at', 'created_at']

    def get_trial_count(self, obj):
        return obj.trials.count()