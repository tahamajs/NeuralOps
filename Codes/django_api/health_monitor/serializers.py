from rest_framework import serializers
from .models import (
    FeatureStore, FeatureGroup, DriftMonitor, DriftCheck,
    Alert, AlertChannel, ModelPerformanceSnapshot
)


class FeatureStoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureStore
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'updated_at']


class FeatureGroupSerializer(serializers.ModelSerializer):
    feature_store_name = serializers.CharField(source='feature_store.name', read_only=True)

    class Meta:
        model = FeatureGroup
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class DriftMonitorSerializer(serializers.ModelSerializer):
    feature_group_name = serializers.CharField(source='feature_group.name', read_only=True)

    class Meta:
        model = DriftMonitor
        fields = '__all__'
        read_only_fields = ['id', 'last_checked_at', 'last_status', 'created_at']


class DriftCheckSerializer(serializers.ModelSerializer):
    monitor_name = serializers.CharField(source='monitor.name', read_only=True)

    class Meta:
        model = DriftCheck
        fields = '__all__'
        read_only_fields = ['id', 'checked_at']


class AlertSerializer(serializers.ModelSerializer):
    monitor_name = serializers.CharField(source='monitor.name', read_only=True)

    class Meta:
        model = Alert
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class AlertChannelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertChannel
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class ModelPerformanceSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelPerformanceSnapshot
        fields = '__all__'
        read_only_fields = ['id', 'recorded_at']