import logging

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from .models import ExperimentLogAnalysis, HPORun, HPOTrial
from .serializers import *
from .tasks import analyze_logs_task, run_hpo_task

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Log Analysis & Anomaly Detection
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([AllowAny])
def anomaly_list(request):
    """List anomalies, optionally filtered by run_id."""
    run_id = request.query_params.get("run_id")
    qs = ExperimentLogAnalysis.objects.all().order_by("-created_at")
    if run_id:
        qs = qs.filter(run_id=run_id)

    anomaly_type = request.query_params.get("type")
    if anomaly_type:
        qs = qs.filter(anomaly_type=anomaly_type)

    min_severity = request.query_params.get("min_severity")
    if min_severity:
        try:
            qs = qs.filter(severity__gte=float(min_severity))
        except ValueError:
            pass

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        ExperimentLogAnalysisSerializer(page, many=True).data
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def trigger_anomaly_detection(request, run_id):
    """Trigger LLM-powered log analysis for a specific run."""
    from experiment.models import TrainingRun

    run = get_object_or_404(TrainingRun, id=run_id)

    # Trigger async task
    analyze_logs_task.delay(str(run_id))
    return Response({"status": "analysis started", "run_id": str(run_id)})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def anomaly_summary(request):
    """Get a summary of all anomalies across all runs."""
    from django.db.models import Count

    qs = (
        ExperimentLogAnalysis.objects.values("anomaly_type")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    return Response(qs)


# ═══════════════════════════════════════════════════════════════
# Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════════
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def hpo_run_list(request):
    """List all HPO runs, or create a new one."""
    if request.method == "GET":
        qs = HPORun.objects.filter(user=request.user).order_by("-created_at")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        return paginator.get_paginated_response(HPORunSerializer(page, many=True).data)

    elif request.method == "POST":
        serializer = HPORunSerializer(data=request.data)
        if serializer.is_valid():
            hpo_run = serializer.save(user=request.user)
            # Trigger async HPO execution
            run_hpo_task.delay(str(hpo_run.id))
            return Response(HPORunSerializer(hpo_run).data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def hpo_run_detail(request, hpo_run_id):
    """Get details or cancel an HPO run."""
    hpo_run = get_object_or_404(HPORun, id=hpo_run_id, user=request.user)

    if request.method == "GET":
        data = HPORunSerializer(hpo_run).data
        data["trials"] = HPOTrialSerializer(
            hpo_run.trials.order_by("trial_number"), many=True
        ).data
        return Response(data)

    elif request.method == "DELETE":
        if hpo_run.status in ("running", "queued"):
            hpo_run.status = "canceled"
            hpo_run.completed_at = timezone.now()
            hpo_run.save()
        return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hpo_trial_detail(request, trial_id):
    """Get details of a specific HPO trial."""
    trial = get_object_or_404(HPOTrial, id=trial_id)
    return Response(HPOTrialSerializer(trial).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hpo_best_result(request, hpo_run_id):
    """Get the best result from an HPO run."""
    hpo_run = get_object_or_404(HPORun, id=hpo_run_id)
    if hpo_run.best_trial:
        return Response(
            {
                "best_objective_value": hpo_run.best_objective_value,
                "best_params": hpo_run.best_params,
                "best_trial_id": str(hpo_run.best_trial.id),
            }
        )
    return Response({"message": "No trials completed yet"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def hpo_search_space_schema(request):
    """Return a template schema for defining search spaces."""
    return Response(
        {
            "examples": [
                {
                    "learning_rate": {
                        "type": "float",
                        "min": 1e-5,
                        "max": 1e-2,
                        "log": True,
                    },
                    "batch_size": {"type": "int", "min": 8, "max": 64},
                    "dropout": {"type": "float", "min": 0.0, "max": 0.5},
                    "attention_type": {
                        "type": "categorical",
                        "choices": ["full", "galerkin", "linformer"],
                    },
                    "use_lora": {"type": "bool"},
                }
            ]
        }
    )
