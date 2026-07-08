from rest_framework import serializers
from .models import (
    ComplianceReport, DataContract, DriftMonitor, DriftAlert,
    SecurityScan, ModelMonitor, ObservationMetric, AccessPolicy, AccessGrant
)
from organizations.serializers import AuditLogSerializer

class ComplianceReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = ComplianceReport
        fields = "__all__"

class DataContractSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataContract
        fields = "__all__"

class DriftMonitorSerializer(serializers.ModelSerializer):
    class Meta:
        model = DriftMonitor
        fields = "__all__"

class DriftAlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = DriftAlert
        fields = "__all__"

class SecurityScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = SecurityScan
        fields = "__all__"

class ModelMonitorSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMonitor
        fields = "__all__"

class ObservationMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = ObservationMetric
        fields = "__all__"

class AccessPolicySerializer(serializers.ModelSerializer):
    class Meta:
        model = AccessPolicy
        fields = "__all__"

class AccessGrantSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccessGrant
        fields = "__all__"
