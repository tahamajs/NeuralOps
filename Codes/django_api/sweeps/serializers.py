from rest_framework import serializers
from experiment.models import Sweep, SweepTrial, TrainingRun

class SweepTrialSerializer(serializers.ModelSerializer):
    run_id = serializers.UUIDField(source='run.id', read_only=True)
    run_name = serializers.CharField(source='run.name', read_only=True)

    class Meta:
        model = SweepTrial
        fields = [
            'id', 'trial_number', 'status', 'objective_value',
            'params', 'run_id', 'run_name', 'created_at',
        ]
        read_only_fields = ['id', 'status', 'objective_value', 'created_at']


class SweepSerializer(serializers.ModelSerializer):
    trials = SweepTrialSerializer(many=True, read_only=True)
    trial_count = serializers.SerializerMethodField()
    best_objective = serializers.SerializerMethodField()

    class Meta:
        model = Sweep
        fields = [
            'id', 'project', 'name', 'config', 'status',
            'direction', 'metric_name', 'trials', 'trial_count',
            'best_objective', 'created_at',
        ]
        read_only_fields = ['id', 'status', 'created_at']

    def get_trial_count(self, obj):
        return obj.trials.count()

    def get_best_objective(self, obj):
        if obj.best_trial:
            return obj.best_trial.objective_value
        # fallback: compute best from trials
        best = obj.trials.exclude(objective_value__isnull=True).order_by(
            f"{'-' if obj.direction == 'maximize' else ''}objective_value"
        ).first()
        return best.objective_value if best else None


class SweepCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sweep
        fields = ['project', 'name', 'config']   # direction, metric inferred from config