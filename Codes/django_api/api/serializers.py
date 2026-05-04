from rest_framework import serializers
from .models import Experiment, TrainingRun, TrainingSnapshot

class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Experiment
        fields = '__all__'

class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'started_at', 'completed_at', 'logs', 'final_metrics')

class TrainingSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingSnapshot
        fields = '__all__'