import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import (PipelineArtifact, PipelineStep, PipelineTemplate,
                     ResearchPipeline)
from .serializers import *
from .tasks import run_pipeline_async

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Research Pipelines
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def pipeline_list(request):
    """List pipelines for the current user, or create a new one."""
    if request.method == "GET":
        qs = ResearchPipeline.objects.filter(user=request.user).order_by("-created_at")
        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ResearchPipelineSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ResearchPipelineSerializer(data=request.data)
        if serializer.is_valid():
            pipeline = serializer.save(user=request.user)
            # If a template_id is provided, populate steps from template
            template_id = request.data.get("template_id")
            if template_id:
                template = get_object_or_404(PipelineTemplate, id=template_id)
                for i, step_def in enumerate(template.steps_config, 1):
                    PipelineStep.objects.create(
                        pipeline=pipeline,
                        step_type=step_def.get("step_type", "literature_search"),
                        order=i,
                        title=step_def.get("title", f"Step {i}"),
                        description=step_def.get("description", ""),
                    )
                template.usage_count += 1
                template.save(update_fields=["usage_count"])
            return Response(ResearchPipelineSerializer(pipeline).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def pipeline_detail(request, pipeline_id):
    """Get, update, or delete a pipeline."""
    pipeline = get_object_or_404(ResearchPipeline, id=pipeline_id, user=request.user)

    if request.method == "GET":
        return Response(ResearchPipelineSerializer(pipeline).data)

    elif request.method == "PATCH":
        serializer = ResearchPipelineSerializer(
            pipeline, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        pipeline.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def launch_pipeline(request, pipeline_id):
    """Launch a pipeline (start the autonomous research process)."""
    pipeline = get_object_or_404(ResearchPipeline, id=pipeline_id, user=request.user)
    if pipeline.status not in ("draft", "paused", "failed"):
        return Response(
            {"error": f"Cannot launch pipeline in status: {pipeline.status}"},
            status=400,
        )

    # Run asynchronously via Celery
    run_pipeline_async.delay(str(pipeline.id))
    return Response({"status": "launched", "pipeline_id": str(pipeline.id)})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def pause_pipeline(request, pipeline_id):
    """Pause a running pipeline."""
    pipeline = get_object_or_404(ResearchPipeline, id=pipeline_id, user=request.user)
    if pipeline.status not in (
        "literature_review",
        "hypothesis_generated",
        "experimenting",
    ):
        return Response({"error": "Pipeline is not running"}, status=400)
    pipeline.status = "paused"
    pipeline.save(update_fields=["status"])
    return Response({"status": "paused"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def resume_pipeline(request, pipeline_id):
    """Resume a paused pipeline."""
    pipeline = get_object_or_404(ResearchPipeline, id=pipeline_id, user=request.user)
    if pipeline.status != "paused":
        return Response({"error": "Pipeline is not paused"}, status=400)
    run_pipeline_async.delay(str(pipeline.id))
    return Response({"status": "resumed"})


# ═══════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def step_list(request, pipeline_id):
    """List all steps in a pipeline."""
    pipeline = get_object_or_404(ResearchPipeline, id=pipeline_id, user=request.user)
    steps = pipeline.steps.order_by("order")
    return Response(PipelineStepSerializer(steps, many=True).data)


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def update_step(request, step_id):
    """Update a step (e.g., approve a human review step)."""
    step = get_object_or_404(PipelineStep, id=step_id, pipeline__user=request.user)
    new_status = request.data.get("status")
    if new_status:
        step.status = new_status
    if "output_data" in request.data:
        step.output_data = request.data["output_data"]
    step.save()
    return Response(PipelineStepSerializer(step).data)


# ═══════════════════════════════════════════════════════════════
# Pipeline Artifacts
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def artifact_list(request, pipeline_id):
    """List all artifacts in a pipeline."""
    pipeline = get_object_or_404(ResearchPipeline, id=pipeline_id, user=request.user)
    artifacts = pipeline.artifacts.order_by("-created_at")
    return Response(PipelineArtifactSerializer(artifacts, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def artifact_detail(request, artifact_id):
    """Get a single artifact with its full content."""
    artifact = get_object_or_404(
        PipelineArtifact, id=artifact_id, pipeline__user=request.user
    )
    return Response(PipelineArtifactSerializer(artifact).data)


# ═══════════════════════════════════════════════════════════════
# Pipeline Templates
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def template_list(request):
    """List public templates, or create a new one."""
    if request.method == "GET":
        qs = PipelineTemplate.objects.filter(is_public=True).order_by("-usage_count")
        category = request.query_params.get("category")
        if category:
            qs = qs.filter(category=category)
        return Response(PipelineTemplateSerializer(qs, many=True).data)

    elif request.method == "POST":
        serializer = PipelineTemplateSerializer(data=request.data)
        if serializer.is_valid():
            template = serializer.save(created_by=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
