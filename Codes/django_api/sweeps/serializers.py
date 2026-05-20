from rest_framework import serializers
from .models import Sweep, SweepTrial

class SweepTrialSerializer(serializers.ModelSerializer):
    run_id = serializers.UUIDField(source='run.id', read_only=True, default=None)
    run_name = serializers.CharField(source='run.name', read_only=True, default=None)
    run_status = serializers.CharField(source='run.status', read_only=True, default=None)

    class Meta:
        model = SweepTrial
        fields = [
            'id', 'trial_number', 'status', 'objective_value', 'objectives',
            'params', 'user_attrs', 'notes', 'is_pareto',
            'run_id', 'run_name', 'run_status',
            'artifacts', 'duration_seconds', 'created_at',
        ]
        read_only_fields = [
            'id', 'status', 'objective_value', 'objectives',
            'is_pareto', 'created_at'
        ]

class SweepSerializer(serializers.ModelSerializer):
    trial_count = serializers.SerializerMethodField()
    best_objective = serializers.SerializerMethodField()
    pareto_count = serializers.SerializerMethodField()
    running_count = serializers.SerializerMethodField()
    completed_count = serializers.SerializerMethodField()

    class Meta:
        model = Sweep
        fields = [
            'id', 'project', 'name', 'description', 'config', 'status',
            'direction', 'directions', 'metric_name', 'metric_names',
            'algorithm', 'pruner', 'scheduler_type',
            'max_trials', 'max_failures', 'parallel_workers',
            'warm_start_from', 'preference_enabled',
            'trial_count', 'best_objective', 'pareto_count',
            'running_count', 'completed_count',
            'tags', 'artifact_dir',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'status', 'created_at', 'updated_at']

    def get_trial_count(self, obj):
        return obj.trials.count()

    def get_best_objective(self, obj):
        if obj.best_trial:
            return obj.best_trial.objective_value
        return None

    def get_pareto_count(self, obj):
        return obj.trials.filter(is_pareto=True).count()

    def get_running_count(self, obj):
        return obj.trials.filter(status='running').count()

    def get_completed_count(self, obj):
        return obj.trials.filter(status='completed').count()

class SweepCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sweep
        fields = [
            'project', 'name', 'description', 'config',
            'direction', 'directions', 'metric_name', 'metric_names',
            'algorithm', 'pruner', 'scheduler_type',
            'max_trials', 'max_failures', 'parallel_workers',
            'warm_start_from', 'preference_enabled',
            'tags', 'artifact_dir',
        ]