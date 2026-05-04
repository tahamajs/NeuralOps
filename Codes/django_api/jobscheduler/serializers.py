from rest_framework import serializers
from .models import Job

class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'started_at', 'completed_at',
                            'logs', 'final_metrics', 'training_history')