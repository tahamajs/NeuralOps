"""
Dataset Management + Artifact Lineage + Audit Logging REST API
"""

import csv
import logging

from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def dataset_list(request):
    if request.method == "GET":
        qs = (
            Dataset.objects.filter(
                Q(is_public=True)
                | Q(owner=request.user)
                | Q(collaborators=request.user)
            )
            .distinct()
            .order_by("-updated_at")
        )

        search = request.query_params.get("q")
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))

        tag = request.query_params.get("tag")
        if tag:
            qs = qs.filter(tags__contains=[tag])

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        serializer = DatasetListSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        serializer = DatasetSerializer(data=request.data)
        if serializer.is_valid():
            dataset = serializer.save(owner=request.user)
            _log_audit(
                request,
                "create",
                "dataset",
                str(dataset.id),
                f"Dataset '{dataset.name}' created",
            )
            return Response(DatasetSerializer(dataset).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def dataset_detail(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if request.method == "GET":
        # Track download/view
        DatasetDownload.objects.create(dataset=dataset, user=request.user)
        _log_audit(request, "read", "dataset", str(dataset_id), f"Dataset viewed")
        return Response(DatasetSerializer(dataset).data)

    elif request.method == "PATCH":
        serializer = DatasetSerializer(dataset, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            _log_audit(
                request, "update", "dataset", str(dataset_id), f"Dataset updated"
            )
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        dataset.delete()
        _log_audit(request, "delete", "dataset", str(dataset_id), f"Dataset deleted")
        return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dataset_export(request, dataset_id):
    """Export dataset metadata as CSV."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{dataset.name}.csv"'
    writer = csv.writer(response)
    writer.writerow(["Version", "Size (MB)", "Rows", "Columns", "Created", "DVC Hash"])
    for v in dataset.versions.all():
        writer.writerow(
            [
                v.version_number,
                round(v.size_bytes / 1e6, 2),
                v.row_count,
                v.column_count,
                v.created_at,
                v.dvc_hash,
            ]
        )
    _log_audit(
        request, "export", "dataset", str(dataset_id), f"Dataset exported as CSV"
    )
    return response


# ═══════════════════════════════════════════════════════════════
# ARTIFACTS & LINEAGE
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def artifact_list(request):
    qs = Artifact.objects.all().order_by("-created_at")
    artifact_type = request.query_params.get("type")
    if artifact_type:
        qs = qs.filter(artifact_type=artifact_type)
    run_id = request.query_params.get("run_id")
    if run_id:
        qs = qs.filter(run_id=run_id)
    search = request.query_params.get("q")
    if search:
        qs = qs.filter(name__icontains=search)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = ArtifactSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def artifact_lineage(request, artifact_id):
    """Return the full lineage graph for an artifact."""
    artifact = get_object_or_404(Artifact, id=artifact_id)
    nodes = {
        str(artifact.id): {
            "id": str(artifact.id),
            "name": artifact.name,
            "type": artifact.artifact_type,
        }
    }
    edges = []

    # Upstream (what this artifact depends on)
    for edge in artifact.upstream_edges.select_related("source"):
        source = edge.source
        nodes[str(source.id)] = {
            "id": str(source.id),
            "name": source.name,
            "type": source.artifact_type,
        }
        edges.append(
            {
                "from": str(source.id),
                "to": str(artifact.id),
                "type": edge.relationship_type,
            }
        )

    # Downstream (what depends on this artifact)
    for edge in artifact.downstream_edges.select_related("target"):
        target = edge.target
        nodes[str(target.id)] = {
            "id": str(target.id),
            "name": target.name,
            "type": target.artifact_type,
        }
        edges.append(
            {
                "from": str(artifact.id),
                "to": str(target.id),
                "type": edge.relationship_type,
            }
        )

    return Response({"nodes": list(nodes.values()), "edges": edges})


# ═══════════════════════════════════════════════════════════════
# AUDIT LOGS (Compliance)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def audit_log_list(request):
    """List audit logs with filtering (staff can see all, users see their own)."""
    if request.user.is_staff:
        qs = AuditLog.objects.all()
    else:
        qs = AuditLog.objects.filter(actor=request.user)

    action_type = request.query_params.get("action_type")
    if action_type:
        qs = qs.filter(action_type=action_type)
    resource_type = request.query_params.get("resource_type")
    if resource_type:
        qs = qs.filter(resource_type=resource_type)
    severity = request.query_params.get("severity")
    if severity:
        qs = qs.filter(severity=severity)
    actor_id = request.query_params.get("actor_id")
    if actor_id and request.user.is_staff:
        qs = qs.filter(actor_id=actor_id)

    qs = qs.order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = AuditLogSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def audit_export_request(request):
    """Request an export of audit logs (staff only or own data)."""
    serializer = AuditExportSerializer(data=request.data)
    if serializer.is_valid():
        export = serializer.save(requested_by=request.user)
        # Trigger Celery task to generate the export
        # generate_audit_export.delay(str(export.id))
        return Response(AuditExportSerializer(export).data, status=201)
    return Response(serializer.errors, status=400)


def _log_audit(
    request,
    action_type,
    resource_type,
    resource_id,
    description,
    data_before=None,
    data_after=None,
    severity="info",
):
    """Utility to create an audit log entry."""
    try:
        AuditLog.objects.create(
            actor=request.user if request.user.is_authenticated else None,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            data_before=data_before or {},
            data_after=data_after or {},
            severity=severity,
            ip_address=request.META.get("REMOTE_ADDR"),
            user_agent=request.META.get("HTTP_USER_AGENT", "")[:500],
        )
    except Exception as e:
        logger.error("Failed to create audit log: %s", e)
