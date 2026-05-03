from rest_framework import serializers
from .models import Experiment

class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Experiment
        fields = "__all__"
        read_only_fields = ["id", "status", "logs", "created_at", "completed_at", "artifacts_dir"]