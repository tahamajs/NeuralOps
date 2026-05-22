import logging
import numpy as np
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from .models import *
from .serializers import *
from .drift_detector import DriftDetector
from .alerting import AlertingEngine

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# FEATURE STORES
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def feature_store_list(request):
    if request.method == 'GET':
        qs = FeatureStore.objects.filter(is_active=True)
        return Response(FeatureStoreSerializer(qs, many=True).data)
    elif request.method == 'POST':
        serializer = FeatureStoreSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# FEATURE GROUPS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def feature_group_list(request):
    if request.method == 'GET':
        qs = FeatureGroup.objects.all()
        store_id = request.query_params.get('store_id')
        if store_id:
            qs = qs.filter(feature_store_id=store_id)
        return Response(FeatureGroupSerializer(qs, many=True).data)
    elif request.method == 'POST':
        serializer = FeatureGroupSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# DRIFT MONITORS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def drift_monitor_list(request):
    if request.method == 'GET':
        qs = DriftMonitor.objects.filter(is_active=True).order_by('-created_at')
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(DriftMonitorSerializer(page, many=True).data)
    elif request.method == 'POST':
        serializer = DriftMonitorSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def run_drift_check(request, monitor_id):
    """
    Manually trigger a drift check for a monitor.
    Body: { "reference_data": [...], "current_data": [...] }
    """
    monitor = get_object_or_404(DriftMonitor, id=monitor_id)

    reference = np.array(request.data.get('reference_data', []))
    current = np.array(request.data.get('current_data', []))

    if len(reference) == 0 or len(current) == 0:
        return Response({"error": "Both reference_data and current_data are required"}, status=400)

    detector = DriftDetector()
    metric_value = detector.detect(monitor.metric, reference, current)

    # Determine status
    if metric_value >= monitor.critical_threshold:
        check_status = 'critical'
    elif metric_value >= monitor.warning_threshold:
        check_status = 'warning'
    else:
        check_status = 'healthy'

    check = DriftCheck.objects.create(
        monitor=monitor,
        metric_value=round(metric_value, 4),
        threshold_value=monitor.warning_threshold,
        status=check_status,
        details={'metric': monitor.metric, 'sample_size': len(current)},
    )

    # Update monitor
    monitor.last_checked_at = timezone.now()
    monitor.last_status = check_status
    monitor.save(update_fields=['last_checked_at', 'last_status'])

    # Trigger alert if needed
    if check_status in ('warning', 'critical'):
        engine = AlertingEngine()
        engine.trigger_alert(monitor, check)

    return Response(DriftCheckSerializer(check).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def drift_check_history(request, monitor_id):
    monitor = get_object_or_404(DriftMonitor, id=monitor_id)
    qs = monitor.checks.order_by('-checked_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(DriftCheckSerializer(page, many=True).data)


# ═══════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def alert_list(request):
    qs = Alert.objects.all().order_by('-created_at')
    unacknowledged = request.query_params.get('unacknowledged', 'false')
    if unacknowledged == 'true':
        qs = qs.filter(is_acknowledged=False)
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(AlertSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def acknowledge_alert(request, alert_id):
    alert = get_object_or_404(Alert, id=alert_id)
    alert.is_acknowledged = True
    alert.acknowledged_by = request.user
    alert.save(update_fields=['is_acknowledged', 'acknowledged_by'])
    return Response({'status': 'acknowledged'})


# ═══════════════════════════════════════════════════════════════
# ALERT CHANNELS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def alert_channel_list(request):
    if request.method == 'GET':
        qs = AlertChannel.objects.filter(user=request.user)
        return Response(AlertChannelSerializer(qs, many=True).data)
    elif request.method == 'POST':
        serializer = AlertChannelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def alert_channel_delete(request, channel_id):
    channel = get_object_or_404(AlertChannel, id=channel_id, user=request.user)
    channel.delete()
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE SNAPSHOTS
# ═══════════════════════════════════════════════════════════════
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def performance_snapshot_list(request):
    if request.method == 'GET':
        qs = ModelPerformanceSnapshot.objects.all().order_by('-recorded_at')
        run_id = request.query_params.get('run_id')
        if run_id:
            qs = qs.filter(training_run_id=run_id)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(ModelPerformanceSnapshotSerializer(page, many=True).data)
    elif request.method == 'POST':
        serializer = ModelPerformanceSnapshotSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)