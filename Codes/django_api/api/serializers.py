# experiments/serializers.py
from rest_framework import serializers
from .models import Experiment, TrainingRun, TrainingSnapshot

class ExperimentSerializer(serializers.ModelSerializer):
    runs_count = serializers.IntegerField(source='runs.count', read_only=True)
    class Meta:
        model = Experiment
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')

class TrainingRunSerializer(serializers.ModelSerializer):
    experiment_name = serializers.CharField(source='experiment.name', read_only=True)
    class Meta:
        model = TrainingRun
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'started_at', 'completed_at', 'logs', 'training_history')

class TrainingSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingSnapshot
        fields = '__all__'
        read_only_fields = ('id', 'timestamp')


# api/serializers.py (or experiment/serializers.py)

class TrainingConfigSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    task_name = serializers.CharField()
    attn_type = serializers.ChoiceField(choices=[])   # populated dynamically
    method = serializers.ChoiceField(choices=[])
    compressor = serializers.CharField(required=False, allow_blank=True)
    trainer_type = serializers.ChoiceField(choices=[])
    loss_type = serializers.ChoiceField(choices=[])
    optimizer_type = serializers.ChoiceField(choices=[])
    selected_callbacks = serializers.ListField(child=serializers.CharField(), required=False)
    epochs = serializers.IntegerField()
    batch_size = serializers.IntegerField()
    learning_rate = serializers.FloatField()
    max_seq_length = serializers.IntegerField()
    # ... all other fields

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamically set choices from registries
        self.fields['attn_type'].choices = list(ATTENTION_REGISTRY.keys())
        self.fields['method'].choices = list(METHOD_REGISTRY.keys())
        self.fields['trainer_type'].choices = list(TRAINER_REGISTRY.keys())
        self.fields['loss_type'].choices = list(LOSS_REGISTRY.keys())
        self.fields['optimizer_type'].choices = list(OPTIMIZER_REGISTRY.keys())