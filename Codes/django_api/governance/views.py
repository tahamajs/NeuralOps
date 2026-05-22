from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db.models import Q, Count
from .models import *
from .serializers import *
from .tasks import run_compliance_check, run_drift_check, run_security_scan

# ── 1. Audit Log Dashboard ─────────────────────────────────────
@api_view(['GET'])
@permission_classes([IsAdminUser])
def audit_log_dashboard(request):
    """Filterable audit log with summary stats."""
    from organizations.models import AuditLog
    qs = AuditLog.objects.select_related('actor', 'organization').order_by('-created_at')

    # Filters
    action = request.query_params.get('action')
    if action: qs = qs.filter(action=action)
    org_id = request.query_params.get('organization_id')
    if org_id: qs = qs.filter(organization_id=org_id)
    user_id = request.query_params.get('user_id')
    if user_id: qs = qs.filter(actor_id=user_id)
    date_from = request.query_params.get('date_from')
    if date_from: qs = qs.filter(created_at__gte=date_from)
    date_to = request.query_params.get('date_to')
    if date_to: qs = qs.filter(created_at__lte=date_to)

    # Export CSV
    if request.query_params.get('export') == 'csv':
        import csv, io
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['Timestamp', 'Actor', 'Action', 'Target User', 'Organization', 'Description'])
        for log in qs[:1000]:
            writer.writerow([log.created_at, log.actor.username if log.actor else '', log.action,
                             log.target_user.username if log.target_user else '', log.organization.name, log.description])
        return HttpResponse(buffer.getvalue(), content_type='text/csv')

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(AuditLogSerializer(page, many=True).data)

# ── 2. Compliance Agent ────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def trigger_compliance_check(request):
    """Run compliance checks on a specific object (run, model, etc.)."""
    content_type = request.data.get('content_type')  # 'experiment.trainingrun'
    object_id = request.data.get('object_id')
    run_compliance_check.delay(content_type, object_id)
    return Response({'status': 'check queued'})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compliance_report(request, report_id):
    report = get_object_or_404(ComplianceReport, id=report_id, user=request.user)
    return Response(ComplianceReportSerializer(report).data)

# ── 3. Data Contracts ──────────────────────────────────────────
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def contract_list(request):
    if request.method == 'GET':
        qs = DataContract.objects.filter(
            Q(provider_org__memberships__user=request.user) | Q(consumer_org__memberships__user=request.user)
        ).distinct()
        return Response(DataContractSerializer(qs, many=True).data)
    elif request.method == 'POST':
        serializer = DataContractSerializer(data=request.data)
        if serializer.is_valid(): serializer.save(); return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def sign_contract(request, contract_id):
    contract = get_object_or_404(DataContract, id=contract_id)
    user_org = request.user.org_memberships.first().organization
    if user_org == contract.provider_org:
        contract.signed_by_provider = True
    elif user_org == contract.consumer_org:
        contract.signed_by_consumer = True
    else:
        return Response({'error': 'Not authorized'}, status=403)
    contract.save()
    return Response({'status': 'signed'})

# ── 4. Drift Detection ─────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def setup_drift_monitor(request, run_id):
    run = get_object_or_404(TrainingRun, id=run_id)
    monitor = DriftMonitor.objects.create(
        run=run,
        feature_columns=request.data.get('feature_columns', []),
        baseline_data_path=request.data.get('baseline_data_path', ''),
        check_interval_hours=request.data.get('check_interval_hours', 24),
    )
    return Response(DriftMonitorSerializer(monitor).data, status=201)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def drift_alerts(request, monitor_id):
    monitor = get_object_or_404(DriftMonitor, id=monitor_id)
    alerts = monitor.alerts.order_by('-created_at')
    return Response(DriftAlertSerializer(alerts, many=True).data)

# ── 5. Security Scanning ───────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def scan_model(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    scan = SecurityScan.objects.create(model_version=version)
    run_security_scan.delay(str(scan.id))
    return Response(SecurityScanSerializer(scan).data, status=201)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def scan_result(request, scan_id):
    scan = get_object_or_404(SecurityScan, id=scan_id)
    return Response(SecurityScanSerializer(scan).data)

# ── 6. Observability ──────────────────────────────────────────
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def setup_model_monitor(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    monitor = ModelMonitor.objects.create(
        model_version=version,
        metrics_config=request.data.get('metrics_config', {}),
    )
    return Response(ModelMonitorSerializer(monitor).data, status=201)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_observations(request, monitor_id):
    monitor = get_object_or_404(ModelMonitor, id=monitor_id)
    hours = int(request.query_params.get('hours', 24))
    since = timezone.now() - timedelta(hours=hours)
    obs = monitor.observations.filter(timestamp__gte=since).order_by('-timestamp')
    return Response(ObservationMetricSerializer(obs, many=True).data)

# ── 7. FGAC ───────────────────────────────────────────────────
@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser])
def access_policy_list(request):
    if request.method == 'GET':
        return Response(AccessPolicySerializer(AccessPolicy.objects.all(), many=True).data)
    elif request.method == 'POST':
        serializer = AccessPolicySerializer(data=request.data)
        if serializer.is_valid(): serializer.save(); return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

@api_view(['POST'])
@permission_classes([IsAdminUser])
def grant_access(request):
    serializer = AccessGrantSerializer(data=request.data)
    if serializer.is_valid(): serializer.save(); return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)