"""Model Hub REST API — complete CRUD, approval workflows, access control, security scans."""

import logging

from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import *
from .serializers import *

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# REGISTERED MODELS — List, Create, Detail, Update, Delete
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def model_list(request):
    if request.method == "GET":
        qs = RegisteredModel.objects.annotate(vcount=Count("versions"))
        # Filters
        stage = request.query_params.get("stage")
        if stage:
            qs = qs.filter(current_stage=stage)
        visibility = request.query_params.get("visibility")
        if visibility:
            qs = qs.filter(visibility=visibility)
        tag = request.query_params.get("tag")
        if tag:
            qs = qs.filter(tags__contains=[tag])
        search = request.query_params.get("q")
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))
        # Show public + user's own
        if not request.user.is_staff:
            qs = qs.filter(
                Q(visibility="public")
                | Q(owner=request.user)
                | Q(collaborators=request.user)
            ).distinct()
        qs = qs.order_by("-updated_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            RegisteredModelSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = RegisteredModelSerializer(data=request.data)
        if serializer.is_valid():
            model = serializer.save(owner=request.user)
            AuditLog.objects.create(
                model=model, action="model_created", actor=request.user
            )
            return Response(RegisteredModelSerializer(model).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def model_detail(request, model_id):
    model = get_object_or_404(RegisteredModel, id=model_id)
    if request.method == "GET":
        return Response(RegisteredModelSerializer(model).data)
    elif request.method == "PATCH":
        if model.owner != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        serializer = RegisteredModelSerializer(model, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            AuditLog.objects.create(
                model=model, action="model_updated", actor=request.user
            )
            return Response(serializer.data)
        return Response(serializer.errors, status=400)
    elif request.method == "DELETE":
        if model.owner != request.user and not request.user.is_staff:
            return Response({"error": "Not allowed"}, status=403)
        model.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# VERSIONS
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_version(request, model_id):
    model = get_object_or_404(RegisteredModel, id=model_id)
    if model.owner != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)

    serializer = ModelVersionSerializer(data=request.data)
    if serializer.is_valid():
        last = model.versions.order_by("-version_number").first()
        next_num = (last.version_number + 1) if last else 1
        version = serializer.save(model=model, version_number=next_num)
        model.version_count = model.versions.count()
        model.save(update_fields=["version_count"])
        AuditLog.objects.create(
            model=model, version=version, action="version_uploaded", actor=request.user
        )
        return Response(ModelVersionSerializer(version).data, status=201)
    return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def version_list(request, model_id):
    model = get_object_or_404(RegisteredModel, id=model_id)
    versions = model.versions.order_by("-version_number")
    return Response(ModelVersionSerializer(versions, many=True).data)


# ═══════════════════════════════════════════════════════════════
# APPROVAL WORKFLOW
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def request_approval(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.model.owner != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)

    req = ApprovalRequest.objects.create(
        version=version,
        request_type="approve",
        requested_by=request.user,
        comment=request.data.get("comment", ""),
    )
    AuditLog.objects.create(
        model=version.model,
        version=version,
        action="approval_requested",
        actor=request.user,
    )
    return Response(ApprovalRequestSerializer(req).data, status=201)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def resolve_approval(request, request_id):
    req = get_object_or_404(ApprovalRequest, id=request_id)
    if not request.user.is_staff:
        return Response({"error": "Only reviewers can resolve"}, status=403)

    resolution = request.data.get("resolution")
    if resolution not in ("approved", "rejected"):
        return Response(
            {"error": "resolution must be 'approved' or 'rejected'"}, status=400
        )

    req.status = resolution
    req.resolved_by = request.user
    req.resolution_comment = request.data.get("comment", "")
    req.resolved_at = timezone.now()
    req.save()

    if resolution == "approved":
        req.version.is_approved = True
        req.version.approved_by = request.user
        req.version.approved_at = timezone.now()
        req.version.save()
        AuditLog.objects.create(
            model=req.version.model,
            version=req.version,
            action="approval_granted",
            actor=request.user,
        )
    else:
        AuditLog.objects.create(
            model=req.version.model,
            version=req.version,
            action="approval_rejected",
            actor=request.user,
        )

    return Response(ApprovalRequestSerializer(req).data)


# ═══════════════════════════════════════════════════════════════
# STAGE PROMOTION
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def promote_version(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    if version.model.owner != request.user and not request.user.is_staff:
        return Response({"error": "Not allowed"}, status=403)

    new_stage = request.data.get("stage")
    if new_stage not in ("staging", "production", "archived"):
        return Response({"error": "Invalid stage"}, status=400)

    old_stage = version.stage
    version.stage = new_stage
    version.save(update_fields=["stage"])

    version.model.current_stage = new_stage
    version.model.save(update_fields=["current_stage"])

    StageTransition.objects.create(
        version=version,
        from_stage=old_stage,
        to_stage=new_stage,
        requested_by=request.user,
        approved_by=request.user,
    )
    AuditLog.objects.create(
        model=version.model,
        version=version,
        action="stage_changed",
        actor=request.user,
        description=f"{old_stage} → {new_stage}",
    )
    return Response(ModelVersionSerializer(version).data)


# ═══════════════════════════════════════════════════════════════
# ACCESS POLICIES
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def access_policy_list(request, model_id):
    model = get_object_or_404(RegisteredModel, id=model_id)

    if request.method == "GET":
        policies = model.access_policies.all()
        return Response(AccessPolicySerializer(policies, many=True).data)

    elif request.method == "POST":
        if model.owner != request.user:
            return Response({"error": "Only owner can manage access"}, status=403)
        serializer = AccessPolicySerializer(data=request.data)
        if serializer.is_valid():
            policy = serializer.save(model=model, granted_by=request.user)
            AuditLog.objects.create(
                model=model,
                action="access_granted",
                actor=request.user,
                description=f"{policy.user or policy.group}: {policy.permission}",
            )
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def revoke_access(request, policy_id):
    policy = get_object_or_404(AccessPolicy, id=policy_id)
    if policy.model.owner != request.user:
        return Response({"error": "Not allowed"}, status=403)
    policy.delete()
    AuditLog.objects.create(
        model=policy.model, action="access_revoked", actor=request.user
    )
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# AUDIT LOG
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def audit_log(request, model_id):
    model = get_object_or_404(RegisteredModel, id=model_id)
    logs = model.audit_logs.order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(logs, request)
    return paginator.get_paginated_response(AuditLogSerializer(page, many=True).data)


# ═══════════════════════════════════════════════════════════════
# CATALOG SEARCH
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def catalog_search(request):
    """Search across all public models with rich filtering."""
    qs = RegisteredModel.objects.filter(visibility="public").annotate(
        vcount=Count("versions")
    )

    search = request.query_params.get("q")
    if search:
        qs = qs.filter(Q(name__icontains=search) | Q(description__icontains=search))

    framework = request.query_params.get("framework")
    if framework:
        qs = qs.filter(framework__icontains=framework)

    task_type = request.query_params.get("task_type")
    if task_type:
        qs = qs.filter(task_type__icontains=task_type)

    tag = request.query_params.get("tag")
    if tag:
        qs = qs.filter(tags__contains=[tag])

    qs = qs.order_by("-download_count")

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        RegisteredModelSerializer(page, many=True).data
    )


# ═══════════════════════════════════════════════════════════════
# RECORD DOWNLOAD
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def record_download(request, version_id):
    version = get_object_or_404(ModelVersion, id=version_id)
    version.model.download_count += 1
    version.model.save(update_fields=["download_count"])

    ModelUsage.objects.create(version=version, user=request.user, usage_type="download")
    AuditLog.objects.create(
        model=version.model,
        version=version,
        action="model_downloaded",
        actor=request.user,
    )
    return Response({"download_count": version.model.download_count})
