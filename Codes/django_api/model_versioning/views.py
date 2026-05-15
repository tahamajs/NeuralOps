
import os
import csv
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
from .models import ModelVersion
from .serializers import ModelVersionSerializer
from .utils import cleanup_old_versions, HuggingFacePublisher

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


# ----------------------------------------------------------------------
# List versions (with pagination, filtering, sorting, CSV export)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_model_versions(request, run_id):
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

    search = request.query_params.get('search')
    if search:
        queryset = queryset.filter(Q(version__icontains=search))

    ordering = request.query_params.get('ordering', '-version_number')
    allowed_orders = ['version_number', 'created_at', 'artifact_size_mb',
                      '-version_number', '-created_at', '-artifact_size_mb']
    if ordering in allowed_orders:
        queryset = queryset.order_by(ordering)
    else:
        queryset = queryset.order_by('-version_number')

    # CSV export
    if request.query_params.get('format') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="model_versions_run_{run.id}.csv"'
        writer = csv.writer(response)
        writer.writerow(['Version Number', 'Version Name', 'Size (MB)', 'Validation Passed',
                         'Published', 'Storage Tier', 'Accuracy', 'Loss', 'Created At'])
        for v in queryset:
            writer.writerow([
                v.version_number, v.version, v.artifact_size_mb,
                v.validation_passed, v.is_published_to_hub, v.storage_tier,
                v.metrics.get('eval_accuracy', 'N/A'),
                v.metrics.get('eval_loss', 'N/A'),
                v.created_at
            ])
        return response

    paginator = StandardPagination()
    page = paginator.paginate_queryset(queryset, request)
    serializer = ModelVersionSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ----------------------------------------------------------------------
# Single version detail
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_version_detail(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    serializer = ModelVersionSerializer(version)
    return Response(serializer.data)


# ----------------------------------------------------------------------
# Publish to Hugging Face Hub (staff only)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def publish_to_hub(request, version_id):
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.is_published_to_hub:
        return Response({"error": "Already published", "hub_url": version.hub_repo_url}, status=400)
    try:
        publisher = HuggingFacePublisher(version)
        hub_url = publisher.publish()
        return Response({'hub_url': hub_url, 'status': 'published'})
    except Exception as e:
        logger.exception(f"Publishing failed for {version_id}")
        return Response({"error": str(e)}, status=500)


# ----------------------------------------------------------------------
# Download model artifact
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_model_version(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    model_file = os.path.join(version.artifact_path, 'model.pth')
    if not os.path.exists(model_file):
        # Try to find any .pth file
        import glob
        pth_files = glob.glob(os.path.join(version.artifact_path, '*.pth'))
        if pth_files:
            model_file = pth_files[0]
        else:
            raise Http404("Model file not found")
    response = FileResponse(open(model_file, 'rb'), as_attachment=True)
    response['Content-Disposition'] = f'attachment; filename="model_v{version.version_number}.pth"'
    return response


# ----------------------------------------------------------------------
# Cleanup old versions (staff only)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cleanup_versions(request, run_id):
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    run = get_object_or_404(Job, id=run_id)
    keep_best = request.data.get('keep_best_k', 3)
    keep_tags = request.data.get('keep_version_tags', ['v1', 'v2'])
    deleted_count = cleanup_old_versions(run, keep_best_k=keep_best, keep_version_tags=keep_tags)
    return Response({"deleted": deleted_count})


# ----------------------------------------------------------------------
# Archive to cold storage (staff only)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def archive_to_cold_storage(request, version_id):
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.storage_tier == 'cold':
        return Response({"error": "Already in cold storage"}, status=400)
    try:
        bucket = getattr(settings, 'MODEL_ARCHIVE_BUCKET', None)
        if not bucket:
            return Response({"error": "MODEL_ARCHIVE_BUCKET not configured"}, status=501)
        version.archive_to_cold_storage(bucket)
        return Response({"status": "archived to cold storage", "storage_tier": "cold"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ----------------------------------------------------------------------
# Restore from cold storage (staff only)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def restore_from_cold_storage(request, version_id):
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.storage_tier != 'cold':
        return Response({"error": "Not in cold storage"}, status=400)
    try:
        bucket = getattr(settings, 'MODEL_ARCHIVE_BUCKET', None)
        if not bucket:
            return Response({"error": "MODEL_ARCHIVE_BUCKET not configured"}, status=501)
        import boto3
        s3 = boto3.client('s3')
        key = f"run_{version.run.id}/v{version.version_number}/model.pth"
        local_path = os.path.join(version.artifact_path, 'model.pth')
        os.makedirs(version.artifact_path, exist_ok=True)
        s3.download_file(bucket, key, local_path)
        version.storage_tier = 'hot'
        version.save(update_fields=['storage_tier'])
        return Response({"status": "restored from cold storage", "storage_tier": "hot"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ----------------------------------------------------------------------
# Set as production version (staff only)
# ----------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def set_production_version(request, run_id, version_number):
    if not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)
    run = get_object_or_404(Job, id=run_id)
    version = get_object_or_404(ModelVersion, run=run, version_number=version_number)
    run.model_versions.filter(is_production=True).update(is_production=False)
    version.is_production = True
    version.save(update_fields=['is_production'])
    return Response({"status": "production version updated", "version_number": version_number})


# ----------------------------------------------------------------------
# Compare two versions
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compare_versions(request, run_id, version_a, version_b):
    run = get_object_or_404(Job, id=run_id)
    v1 = get_object_or_404(ModelVersion, run=run, version_number=version_a)
    v2 = get_object_or_404(ModelVersion, run=run, version_number=version_b)
    metrics_diff = {}
    all_keys = set(v1.metrics.keys()) | set(v2.metrics.keys())
    for key in all_keys:
        val1 = v1.metrics.get(key)
        val2 = v2.metrics.get(key)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            metrics_diff[key] = {"v1": val1, "v2": val2, "diff": val2 - val1}
        else:
            metrics_diff[key] = {"v1": val1, "v2": val2}
    return Response({
        "version_a": {"number": v1.version_number, "created_at": v1.created_at, "size_mb": v1.artifact_size_mb},
        "version_b": {"number": v2.version_number, "created_at": v2.created_at, "size_mb": v2.artifact_size_mb},
        "metrics_diff": metrics_diff,
        "size_diff_mb": round(v2.artifact_size_mb - v1.artifact_size_mb, 2)
    })


# ----------------------------------------------------------------------
# All model versions (recent 50)
# ----------------------------------------------------------------------
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def all_model_versions(request):
    versions = ModelVersion.objects.select_related('run').order_by('-created_at')[:50]
    data = [{
        'id': str(v.id),
        'run_id': str(v.run_id),
        'run_name': v.run.name,
        'version': v.version,
        'metrics': v.metrics,
        'artifact_size_mb': v.artifact_size_mb,
        'is_published_to_hub': v.is_published_to_hub,
        'created_at': v.created_at.isoformat(),
    } for v in versions]
    return Response(data)