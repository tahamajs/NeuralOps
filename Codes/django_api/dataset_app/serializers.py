from rest_framework import serializers
from .models import Dataset, Artifact, AuditLog, AuditExport

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = "__all__"

class DatasetListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ["id", "name", "description", "status", "created_at", "updated_at"]

class ArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Artifact
        fields = "__all__"

class AuditLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLog
        fields = "__all__"

class AuditExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditExport
        fields = "__all__"
