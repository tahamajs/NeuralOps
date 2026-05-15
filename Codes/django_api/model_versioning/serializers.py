from rest_framework import serializers
from .models import ModelVersion

class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
