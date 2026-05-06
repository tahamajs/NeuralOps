from rest_framework import serializers
from .models import Job, CustomMetricEntry, VersionedConfig

class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'started_at', 'completed_at',
                            'logs', 'final_metrics', 'training_history')

class CustomMetricEntrySerializer(serializers.ModelSerializer):
    run = serializers.PrimaryKeyRelatedField(queryset=Job.objects.all(), required=False)

    class Meta:
        model = CustomMetricEntry
        fields = '__all__'
        read_only_fields = ('id', 'created_at')

class VersionedConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = VersionedConfig
        fields = '__all__'
        read_only_fields = ('id', 'created_at')