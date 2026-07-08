from rest_framework import serializers
from integrations.models import (
    CloudConnection, SyncTask, PluginDefinition, PluginActivation,
    SandboxSession, SandboxAuditLog, WebhookSubscription, WebhookDelivery
)

class CloudConnectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CloudConnection
        fields = "__all__"

class SyncTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = SyncTask
        fields = "__all__"

class PluginDefinitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PluginDefinition
        fields = "__all__"

class PluginActivationSerializer(serializers.ModelSerializer):
    class Meta:
        model = PluginActivation
        fields = "__all__"

class SandboxSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SandboxSession
        fields = "__all__"

class SandboxAuditLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = SandboxAuditLog
        fields = "__all__"

class WebhookSubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebhookSubscription
        fields = "__all__"

class WebhookDeliverySerializer(serializers.ModelSerializer):
    class Meta:
        model = WebhookDelivery
        fields = "__all__"
