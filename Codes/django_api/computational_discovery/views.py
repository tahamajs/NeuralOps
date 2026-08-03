import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import CodeExperiment, CodeVariant, ExperimentRun
from .serializers import (
    CodeExperimentSerializer,
    CodeVariantSerializer,
    ExperimentRunSerializer,
)
from .tasks import run_computational_experiment

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Code Experiments
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def experiment_list(request):
    """List all computational experiments, or create a new one."""
    if request.method == "GET":
        qs = CodeExperiment.objects.filter(owner=request.user).order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            CodeExperimentSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = CodeExperimentSerializer(data=request.data)
        if serializer.is_valid():
            experiment = serializer.save(owner=request.user)
            return Response(CodeExperimentSerializer(experiment).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "PATCH", "DELETE"])
@permission_classes([IsAuthenticated])
def experiment_detail(request, experiment_id):
    """Get, update, or delete a computational experiment."""
    experiment = get_object_or_404(CodeExperiment, id=experiment_id, owner=request.user)

    if request.method == "GET":
        return Response(CodeExperimentSerializer(experiment).data)

    elif request.method == "PATCH":
        serializer = CodeExperimentSerializer(
            experiment, data=request.data, partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=400)

    elif request.method == "DELETE":
        experiment.delete()
        return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Launch Experiment (start the auto-evolution loop)
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def launch_experiment(request, experiment_id):
    """Launch a computational experiment – start generating and testing variants."""
    experiment = get_object_or_404(CodeExperiment, id=experiment_id, owner=request.user)

    if experiment.status not in ("draft", "paused"):
        return Response(
            {"error": f"Cannot launch in status: {experiment.status}"}, status=400
        )

    experiment.status = "generating"
    experiment.save(update_fields=["status"])

    # Trigger async task
    run_computational_experiment.delay(str(experiment.id))

    return Response({"status": "launched", "experiment_id": str(experiment.id)})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def pause_experiment(request, experiment_id):
    """Pause a running experiment."""
    experiment = get_object_or_404(CodeExperiment, id=experiment_id, owner=request.user)
    if experiment.status in ("generating", "testing"):
        experiment.status = "paused"
        experiment.save(update_fields=["status"])
        return Response({"status": "paused"})
    return Response({"error": "Cannot pause in current status"}, status=400)


# ═══════════════════════════════════════════════════════════════
# Variants
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def variant_list(request, experiment_id):
    """List all variants of an experiment."""
    experiment = get_object_or_404(CodeExperiment, id=experiment_id, owner=request.user)
    qs = experiment.variants.order_by("-score", "-version_number")

    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter)

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(CodeVariantSerializer(page, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def variant_detail(request, variant_id):
    """Get a single code variant."""
    variant = get_object_or_404(CodeVariant, id=variant_id)
    if variant.experiment.owner != request.user:
        return Response({"error": "Not allowed"}, status=403)
    return Response(CodeVariantSerializer(variant).data)


# ═══════════════════════════════════════════════════════════════
# Experiment Runs
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_list(request, variant_id):
    """List all test runs for a variant."""
    variant = get_object_or_404(CodeVariant, id=variant_id)
    if variant.experiment.owner != request.user:
        return Response({"error": "Not allowed"}, status=403)

    qs = variant.runs.order_by("-created_at")
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ExperimentRunSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_detail(request, run_id):
    """Get a single experiment run."""
    run = get_object_or_404(ExperimentRun, id=run_id)
    if run.variant.experiment.owner != request.user:
        return Response({"error": "Not allowed"}, status=403)
    return Response(ExperimentRunSerializer(run).data)


# ═══════════════════════════════════════════════════════════════
# Leaderboard (across all experiments)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def global_leaderboard(request):
    """Global leaderboard of best-performing variants across all experiments."""
    qs = (
        CodeVariant.objects.filter(status="completed", score__isnull=False)
        .select_related("experiment", "created_by")
        .order_by("-score")[:50]
    )

    data = [
        {
            "rank": i + 1,
            "score": v.score,
            "experiment_name": v.experiment.name,
            "variant_version": v.version_number,
            "created_by": v.created_by.username,
            "metrics": v.metrics,
        }
        for i, v in enumerate(qs)
    ]

    return Response(data)
