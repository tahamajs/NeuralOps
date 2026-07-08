import csv
import io
import logging

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import ExperimentDataset, SavedVisualization
from .serializers import (ExperimentDatasetSerializer,
                          SavedVisualizationSerializer)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Experiment Datasets
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def dataset_list(request):
    """List my datasets or upload a new one."""
    if request.method == "GET":
        qs = ExperimentDataset.objects.filter(user=request.user).order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(
            ExperimentDatasetSerializer(page, many=True).data
        )

    elif request.method == "POST":
        serializer = ExperimentDatasetSerializer(data=request.data)
        if serializer.is_valid():
            dataset = serializer.save(user=request.user)
            return Response(ExperimentDatasetSerializer(dataset).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def dataset_detail(request, dataset_id):
    """Get or delete a dataset."""
    dataset = get_object_or_404(ExperimentDataset, id=dataset_id, user=request.user)
    if request.method == "GET":
        return Response(ExperimentDatasetSerializer(dataset).data)
    elif request.method == "DELETE":
        dataset.delete()
        return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_csv(request):
    """Upload a CSV file to create a new experiment dataset."""
    file = request.FILES.get("file")
    if not file:
        return Response({"error": "file required"}, status=400)

    decoded = file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    data = []
    for row in reader:
        # Try to convert numeric values
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
        data.append(row)

    dataset = ExperimentDataset.objects.create(
        user=request.user,
        name=request.data.get("name", file.name),
        data=data,
    )
    return Response(ExperimentDatasetSerializer(dataset).data, status=201)


# ═══════════════════════════════════════════════════════════════
# Saved Visualizations
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def visualization_list(request, dataset_id):
    """List visualizations for a dataset or save a new one."""
    dataset = get_object_or_404(ExperimentDataset, id=dataset_id, user=request.user)

    if request.method == "GET":
        qs = dataset.visualizations.all().order_by("-created_at")
        return Response(SavedVisualizationSerializer(qs, many=True).data)

    elif request.method == "POST":
        serializer = SavedVisualizationSerializer(data=request.data)
        if serializer.is_valid():
            viz = serializer.save(user=request.user, dataset=dataset)
            return Response(SavedVisualizationSerializer(viz).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def visualization_delete(request, viz_id):
    """Delete a saved visualization."""
    viz = get_object_or_404(SavedVisualization, id=viz_id, user=request.user)
    viz.delete()
    return Response(status=204)


# ═══════════════════════════════════════════════════════════════
# Auto-import from Training Runs (bonus)
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def import_from_training_runs(request):
    """Create a dataset from the user's completed training runs."""
    from experiment.models import TrainingRun

    run_ids = request.data.get("run_ids", [])
    runs = TrainingRun.objects.filter(
        id__in=run_ids, user=request.user, status="completed"
    )

    data = []
    for run in runs:
        row = run.config.copy() if run.config else {}
        if run.final_metrics:
            row.update(run.final_metrics)
        data.append(row)

    dataset = ExperimentDataset.objects.create(
        user=request.user,
        name=request.data.get("name", "Imported from Training Runs"),
        data=data,
    )
    return Response(ExperimentDatasetSerializer(dataset).data, status=201)
