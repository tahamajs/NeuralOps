"""
Complete views for the model versioning app.
Covers version listing, detail, diffing, reconstruction, compression catalog,
lineage graph, lifecycle migration, download, publish, archive/restore, cleanup,
and production flagging.
"""

import os
import csv
import io
import tarfile
import tempfile
import logging
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404
from django.db.models import Q
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

from jobscheduler.models import Job
from .models import ModelVersion, ModelVersionDiff
from .serializers import ModelVersionSerializer
from .utils import cleanup_old_versions, HuggingFacePublisher
from .services import create_diff, load_version_from_diff, migrate_storage_tier

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════════
# Global version listing (recent 50)
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def all_model_versions(request):
    """Return the most recent 50 model versions across all runs."""
    versions = ModelVersion.objects.select_related('run').order_by('-created_at')[:50]
    data = [{
        'id': str(v.id),
        'run_id': str(v.run_id),
        'run_name': v.run.name,
        'version': v.version,
        'version_number': v.version_number,
        'metrics': v.metrics,
        'artifact_size_mb': v.artifact_size_mb,
        'compressed_size_mb': v.compressed_size_mb,
        'is_published_to_hub': v.is_published_to_hub,
        'is_production': v.is_production,
        'storage_tier': v.storage_tier,
        'created_at': v.created_at.isoformat(),
    } for v in versions]
    return Response(data)


# ═══════════════════════════════════════════════════════════════════
# Per‑run version listing (paginated, filterable, CSV export)
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_model_versions(request, run_id):
    """List versions for a specific run with optional filtering, sorting, and CSV export."""
    run = get_object_or_404(Job, id=run_id)
    queryset = run.model_versions.all().order_by('-version_number')

    # Filtering
    validation = request.query_params.get('validation_passed')
    if validation is not None:
        queryset = queryset.filter(validation_passed=validation.lower() == 'true')

    published = request.query_params.get('published')
    if published is not None:
        queryset = queryset.filter(is_published_to_hub=published.lower() == 'true')

    storage_tier = request.query_params.get('storage_tier')
    if storage_tier in ['hot', 'warm', 'cold']:
        queryset = queryset.filter(storage_tier=storage_tier)

    compression_method = request.query_params.get('compression_method')
    if compression_method:
        queryset = queryset.filter(compression_method=compression_method)

    search = request.query_params.get('search')
    if search:
        queryset = queryset.filter(Q(version__icontains=search))

    ordering = request.query_params.get('ordering', '-version_number')
    allowed_orders = [
        'version_number', 'created_at', 'artifact_size_mb',
        'compressed_size_mb',
        '-version_number', '-created_at', '-artifact_size_mb',
        '-compressed_size_mb'
    ]
    if ordering in allowed_orders:
        queryset = queryset.order_by(ordering)
    else:
        queryset = queryset.order_by('-version_number')

    # CSV export
    if request.query_params.get('format') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="model_versions_run_{run.id}.csv"'
        writer = csv.writer(response)
        writer.writerow([
            'Version Number', 'Version Name', 'Size (MB)', 'Compressed Size (MB)',
            'Compression Method', 'Validation Passed', 'Published', 'Storage Tier',
            'Accuracy', 'Loss', 'Created At'
        ])
        for v in queryset:
            writer.writerow([
                v.version_number, v.version, v.artifact_size_mb, v.compressed_size_mb,
                v.compression_method or 'none', v.validation_passed,
                v.is_published_to_hub, v.storage_tier,
                v.metrics.get('eval_accuracy', 'N/A'),
                v.metrics.get('eval_loss', 'N/A'),
                v.created_at
            ])
        return response

    paginator = StandardPagination()
    page = paginator.paginate_queryset(queryset, request)
    serializer = ModelVersionSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ═══════════════════════════════════════════════════════════════════
# Single version detail
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_version_detail(request, version_id):
    """Return detailed information about a single model version."""
    version = get_object_or_404(ModelVersion, id=version_id)
    serializer = ModelVersionSerializer(version)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════════
# Publish to Hugging Face Hub (staff only)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def publish_to_hub(request, version_id):
    """Publish a model version to Hugging Face Hub. Staff only."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.is_published_to_hub:
        return Response(
            {"error": "Already published", "hub_url": version.hub_repo_url},
            status=400
        )
    try:
        publisher = HuggingFacePublisher(version)
        hub_url = publisher.publish()
        return Response({'hub_url': hub_url, 'status': 'published'})
    except Exception as e:
        logger.exception(f"Publishing failed for {version_id}")
        return Response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
# Download model artifact
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_model_version(request, version_id):
    """
    Download a model version. Prefers the compressed artifact if available,
    otherwise streams the artifact directory as a tar.gz.
    """
    version = get_object_or_404(ModelVersion, id=version_id)

    # 1. Prefer compressed artifact
    if version.compressed_artifact_path and os.path.exists(version.compressed_artifact_path):
        response = FileResponse(
            open(version.compressed_artifact_path, 'rb'),
            as_attachment=True,
            content_type='application/gzip'
        )
        ext = os.path.splitext(version.compressed_artifact_path)[1] or '.tar.gz'
        response['Content-Disposition'] = (
            f'attachment; filename="model_v{version.version_number}{ext}"'
        )
        return response

    # 2. If no compressed artifact, stream the artifact directory as tar.gz
    if not os.path.isdir(version.artifact_path):
        raise Http404("Artifact not found")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        tar.add(version.artifact_path, arcname=os.path.basename(version.artifact_path))
    tar_buffer.seek(0)
    response = FileResponse(tar_buffer, as_attachment=True, content_type='application/gzip')
    response['Content-Disposition'] = (
        f'attachment; filename="model_v{version.version_number}.tar.gz"'
    )
    return response


# ═══════════════════════════════════════════════════════════════════
# Cleanup old versions (staff only)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cleanup_versions(request, run_id):
    """Delete old versions, keeping the best K and any tagged versions. Staff only."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    run = get_object_or_404(Job, id=run_id)
    keep_best = int(request.data.get('keep_best_k', 3))
    keep_tags = request.data.get('keep_version_tags', [])
    deleted_count = cleanup_old_versions(
        run, keep_best_k=keep_best, keep_version_tags=keep_tags
    )
    return Response({"deleted": deleted_count})


# ═══════════════════════════════════════════════════════════════════
# Archive to cold storage (staff only)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def archive_to_cold_storage(request, version_id):
    """Move a version's artifact to cold storage (S3 Glacier). Staff only."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.storage_tier == 'cold':
        return Response({"error": "Already in cold storage"}, status=400)
    bucket = getattr(settings, 'MODEL_ARCHIVE_BUCKET', None)
    if not bucket:
        return Response({"error": "MODEL_ARCHIVE_BUCKET not configured"}, status=501)
    try:
        version.archive_to_cold_storage(bucket)
        return Response({
            "status": "archived to cold storage",
            "storage_tier": version.storage_tier
        })
    except Exception as e:
        logger.exception(f"Archive failed for {version_id}")
        return Response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
# Restore from cold storage (staff only)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def restore_from_cold_storage(request, version_id):
    """Restore a version's artifact from cold storage. Staff only."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.storage_tier != 'cold':
        return Response({"error": "Not in cold storage"}, status=400)
    bucket = getattr(settings, 'MODEL_ARCHIVE_BUCKET', None)
    if not bucket:
        return Response({"error": "MODEL_ARCHIVE_BUCKET not configured"}, status=501)
    try:
        import boto3
        s3 = boto3.client('s3')
        key = f"run_{version.run.id}/v{version.version_number}/model.pth"
        local_path = os.path.join(version.artifact_path, 'model.pth')
        os.makedirs(version.artifact_path, exist_ok=True)
        s3.download_file(bucket, key, local_path)
        version.storage_tier = 'hot'
        version.save(update_fields=['storage_tier'])
        return Response({
            "status": "restored from cold storage",
            "storage_tier": "hot"
        })
    except Exception as e:
        logger.exception(f"Restore failed for {version_id}")
        return Response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
# Set as production version (staff only)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def set_production_version(request, run_id, version_number):
    """Mark a specific version as the production model for a run. Staff only."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    run = get_object_or_404(Job, id=run_id)
    version = get_object_or_404(ModelVersion, run=run, version_number=version_number)
    # Unset previous production flag
    run.model_versions.filter(is_production=True).update(is_production=False)
    version.is_production = True
    version.save(update_fields=['is_production'])
    return Response({
        "status": "production version updated",
        "version_number": version_number
    })


# ═══════════════════════════════════════════════════════════════════
# Compare two versions
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compare_versions(request, run_id, version_a, version_b):
    """Compare metrics and sizes between two versions of the same run."""
    run = get_object_or_404(Job, id=run_id)
    v1 = get_object_or_404(ModelVersion, run=run, version_number=version_a)
    v2 = get_object_or_404(ModelVersion, run=run, version_number=version_b)

    metrics_diff = {}
    all_keys = set(v1.metrics.keys()) | set(v2.metrics.keys())
    for key in all_keys:
        val1 = v1.metrics.get(key)
        val2 = v2.metrics.get(key)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            metrics_diff[key] = {"v1": val1, "v2": val2, "diff": round(val2 - val1, 6)}
        else:
            metrics_diff[key] = {"v1": val1, "v2": val2}

    return Response({
        "version_a": {
            "id": str(v1.id),
            "version_number": v1.version_number,
            "version": v1.version,
            "created_at": v1.created_at,
            "size_mb": v1.artifact_size_mb,
            "compressed_size_mb": v1.compressed_size_mb,
            "compression_method": v1.compression_method,
        },
        "version_b": {
            "id": str(v2.id),
            "version_number": v2.version_number,
            "version": v2.version,
            "created_at": v2.created_at,
            "size_mb": v2.artifact_size_mb,
            "compressed_size_mb": v2.compressed_size_mb,
            "compression_method": v2.compression_method,
        },
        "metrics_diff": metrics_diff,
        "size_diff_mb": round(v2.artifact_size_mb - v1.artifact_size_mb, 2),
        "compressed_size_diff_mb": round(
            (v2.compressed_size_mb or 0) - (v1.compressed_size_mb or 0), 2
        )
    })


# ═══════════════════════════════════════════════════════════════════
# Diff creation and reconstruction
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_diff_view(request, from_version_id, to_version_id):
    """Create a diff between two versions of the same run."""
    v1 = get_object_or_404(ModelVersion, id=from_version_id)
    v2 = get_object_or_404(ModelVersion, id=to_version_id)
    if v1.run_id != v2.run_id:
        return Response(
            {"error": "Versions must belong to the same run"},
            status=400
        )
    try:
        diff = create_diff(v1, v2)
        return Response({
            "diff_id": str(diff.id),
            "diff_size_mb": diff.diff_size_mb,
            "original_size_saved_mb": diff.original_size_saved_mb
        }, status=201)
    except Exception as e:
        logger.exception("Diff creation failed")
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def reconstruct_version(request, base_version_id, diff_id):
    """Reconstruct a model version by applying a diff to a base version."""
    base = get_object_or_404(ModelVersion, id=base_version_id)
    diff = get_object_or_404(ModelVersionDiff, id=diff_id)
    try:
        model = load_version_from_diff(base, diff)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
                tar.add(tmpdir, arcname='reconstructed')
            tar_buffer.seek(0)
            response = FileResponse(
                tar_buffer,
                as_attachment=True,
                content_type='application/gzip'
            )
            response['Content-Disposition'] = (
                f'attachment; filename="reconstructed_v{base.version_number}_from_diff.tar.gz"'
            )
            return response
    except Exception as e:
        logger.exception("Reconstruction failed")
        return Response({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
# Lineage graph
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def lineage_graph(request, version_id):
    """Return the full lineage graph for a model version."""
    from .models import get_lineage_graph
    direction = request.query_params.get('direction', 'both')
    max_depth = int(request.query_params.get('max_depth', 5))
    graph = get_lineage_graph(str(version_id), direction=direction, max_depth=max_depth)
    return Response(graph)


# ═══════════════════════════════════════════════════════════════════
# Compression methods catalog
# ═══════════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_compression_methods(request):
    """List available compression methods from the framework."""
    try:
        from framework.core.registry import COMPRESSOR_REGISTRY
        methods = {
            'none': 'No compression',
            'gzip': 'Gzip tar archive',
        }
        for key in COMPRESSOR_REGISTRY.keys():
            methods[key] = f'Framework compressor: {key}'
        return Response(methods)
    except ImportError:
        return Response({
            'none': 'No compression',
            'gzip': 'Gzip tar archive',
        })


# ═══════════════════════════════════════════════════════════════════
# Lifecycle migration (staff only)
# ═══════════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def run_lifecycle_migration(request):
    """Manually trigger storage tier migration for all versions. Staff only."""
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    try:
        for version in ModelVersion.objects.all():
            migrate_storage_tier(version)
        return Response({"status": "lifecycle migration completed"})
    except Exception as e:
        logger.exception("Lifecycle migration failed")
        return Response({"error": str(e)}, status=500)
    

# ═══════════════════════════════════════════════════════════════
# GOVERNANCE — Approval Workflow
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def request_approval(request, version_id):
    """Submit a model version for approval."""
    version = get_object_or_404(ModelVersion, id=version_id)

    if ModelApproval.objects.filter(version=version).exists():
        return Response({"error": "Approval already requested for this version"}, status=400)

    approval = ModelApproval.objects.create(
        version=version,
        requested_by=request.user,
        notes=request.data.get('notes', ''),
    )
    return Response(ModelApprovalSerializer(approval).data, status=201)


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def review_approval(request, approval_id):
    """Approve or reject a model version (staff / reviewer)."""
    approval = get_object_or_404(ModelApproval, id=approval_id)
    action = request.data.get('action')

    if action == 'approve':
        approval.approve(request.user, request.data.get('notes', ''))
    elif action == 'reject':
        approval.reject(request.user, request.data.get('notes', ''))
    else:
        return Response({"error": "action must be 'approve' or 'reject'"}, status=400)

    return Response(ModelApprovalSerializer(approval).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def pending_approvals(request):
    """List all pending approval requests."""
    qs = ModelApproval.objects.filter(status='pending').order_by('-created_at')
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ModelApprovalSerializer(page, many=True).data)


# ═══════════════════════════════════════════════════════════════
# GOVERNANCE — Access Control
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def grant_access(request, version_id):
    """Grant access to a model version."""
    version = get_object_or_404(ModelVersion, id=version_id)
    user_id = request.data.get('user_id')
    access_level = request.data.get('access_level', 'read')

    user = get_object_or_404(User, id=user_id)

    access, created = ModelAccessControl.objects.update_or_create(
        version=version, user=user,
        defaults={
            'access_level': access_level,
            'granted_by': request.user,
        }
    )
    return Response(ModelAccessControlSerializer(access).data,
                    status=201 if created else 200)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def revoke_access(request, access_id):
    """Revoke access to a model version."""
    access = get_object_or_404(ModelAccessControl, id=access_id)
    access.delete()
    return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def version_access_list(request, version_id):
    """List all access grants for a model version."""
    version = get_object_or_404(ModelVersion, id=version_id)
    qs = version.access_controls.select_related('user', 'granted_by').order_by('-granted_at')
    return Response(ModelAccessControlSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# GOVERNANCE — Security Scans
# ═══════════════════════════════════════════════════════════════
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def trigger_security_scan(request, version_id):
    """Trigger a security scan for a model version."""
    version = get_object_or_404(ModelVersion, id=version_id)
    scan_type = request.data.get('scan_type', 'vulnerability')

    scan = ModelSecurityScan.objects.create(
        version=version,
        scan_type=scan_type,
        status='pending',
    )

    # In production, trigger a Celery task to run the actual scan
    from .tasks import run_security_scan
    run_security_scan.delay(str(scan.id))

    return Response(ModelSecurityScanSerializer(scan).data, status=201)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def security_scan_results(request, version_id):
    """List security scan results for a model version."""
    version = get_object_or_404(ModelVersion, id=version_id)
    qs = version.security_scans.order_by('-scanned_at')
    return Response(ModelSecurityScanSerializer(qs, many=True).data)


# ═══════════════════════════════════════════════════════════════
# GOVERNANCE — Usage Audit Log
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def usage_log(request, version_id):
    """Get usage audit logs for a model version."""
    version = get_object_or_404(ModelVersion, id=version_id)
    qs = version.usage_logs.select_related('user').order_by('-created_at')

    action = request.query_params.get('action')
    if action:
        qs = qs.filter(action=action)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(ModelUsageLogSerializer(page, many=True).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def log_usage(request, version_id):
    """Log a model usage event."""
    version = get_object_or_404(ModelVersion, id=version_id)
    action = request.data.get('action', 'viewed')

    log_entry = ModelUsageLog.objects.create(
        version=version,
        user=request.user,
        action=action,
        metadata=request.data.get('metadata', {}),
        ip_address=request.META.get('REMOTE_ADDR'),
    )
    return Response(ModelUsageLogSerializer(log_entry).data, status=201)


# ═══════════════════════════════════════════════════════════════
# GOVERNANCE — Dashboard
# ═══════════════════════════════════════════════════════════════
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def governance_dashboard(request):
    """Governance dashboard showing all models, approvals, and security status."""
    total_models = ModelVersion.objects.count()
    pending = ModelApproval.objects.filter(status='pending').count()
    approved = ModelApproval.objects.filter(status='approved').count()
    rejected = ModelApproval.objects.filter(status='rejected').count()
    high_severity = ModelSecurityScan.objects.filter(severity__in=['high', 'critical']).count()
    recent_usage = ModelUsageLog.objects.order_by('-created_at')[:20]

    return Response({
        'total_models': total_models,
        'approvals': {
            'pending': pending,
            'approved': approved,
            'rejected': rejected,
        },
        'high_severity_issues': high_severity,
        'recent_usage': ModelUsageLogSerializer(recent_usage, many=True).data,
    })