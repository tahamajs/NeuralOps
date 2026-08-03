from rest_framework import serializers

from .models import (
    AccessPolicy,
    ApprovalRequest,
    AuditLog,
    ModelVersion,
    RegisteredModel,
)


class RegisteredModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegisteredModel
        fields = "__all__"


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = "__all__"


class ApprovalRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApprovalRequest
        fields = "__all__"


class AccessPolicySerializer(serializers.ModelSerializer):
    class Meta:
        model = AccessPolicy
        fields = "__all__"


class AuditLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLog
        fields = "__all__"
