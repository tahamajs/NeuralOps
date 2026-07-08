from rest_framework import serializers
from .models import InferenceEndpoint, PipelineRun, DistillationJob, MultiNodeTraining

class InferenceEndpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = InferenceEndpoint
        fields = "__all__"

class PipelineRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = PipelineRun
        fields = "__all__"

class DistillationJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = DistillationJob
        fields = "__all__"

class MultiNodeTrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = MultiNodeTraining
        fields = "__all__"
