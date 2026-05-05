# model_versioning/serializers.py
from rest_framework import serializers
from .models import ModelVersion, ModelVersionDiff

class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')

class ModelVersionDiffSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersionDiff
        fields = '__all__'
        read_only_fields = ('id', 'created_at')