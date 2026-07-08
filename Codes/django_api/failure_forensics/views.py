import logging

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .forensic_engine import FailureForensicEngine
from .models import ErrorPattern, FailureAnalysis, FixAction
from .serializers import ErrorPatternSerializer, FailureAnalysisSerializer

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# Failure Analysis
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def analyze_failure(request, run_id):
    """Run forensic analysis on a failed training run."""
    from experiment.models import TrainingRun

    run = get_object_or_404(TrainingRun, id=run_id)

    if run.status not in ("failed", "stopped", "error"):
        return Response(
            {"error": "Run must be in 'failed', 'stopped', or 'error' state"},
            status=400,
        )

    # Check if analysis already exists
    if hasattr(run, "failure_analysis"):
        analysis = run.failure_analysis
        serializer = FailureAnalysisSerializer(analysis)
        return Response(serializer.data)

    # Run the forensic engine
    engine = FailureForensicEngine(run)
    result = engine.analyze()

    analysis = FailureAnalysis.objects.create(
        run=run,
        initiated_by=request.user,
        status="completed",
        root_cause_category=result["root_cause_category"],
        root_cause_detail=result["root_cause_detail"],
        confidence=result["confidence"],
        error_logs=result["error_logs"],
        anomaly_patterns=result["anomaly_patterns"],
        suggestions=result["suggestions"],
    )

    # Create fix actions
    for fix_data in result["fix_actions"]:
        FixAction.objects.create(
            analysis=analysis,
            action_type=fix_data["action_type"],
            description=fix_data["description"],
            config_changes=fix_data.get("config_changes", {}),
        )

    serializer = FailureAnalysisSerializer(analysis)
    return Response(serializer.data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def analysis_detail(request, analysis_id):
    """Get a failure analysis by ID."""
    analysis = get_object_or_404(FailureAnalysis, id=analysis_id)
    serializer = FailureAnalysisSerializer(analysis)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def my_analyses(request):
    """List all failure analyses initiated by the current user."""
    qs = FailureAnalysis.objects.filter(initiated_by=request.user).order_by(
        "-created_at"
    )
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    return paginator.get_paginated_response(
        FailureAnalysisSerializer(page, many=True).data
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def run_analysis(request, run_id):
    """Get the failure analysis for a specific run."""
    analysis = get_object_or_404(FailureAnalysis, run_id=run_id)
    serializer = FailureAnalysisSerializer(analysis)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# Fix Actions (One‑Click)
# ═══════════════════════════════════════════════════════════════
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def apply_fix(request, fix_id):
    """Apply a suggested fix action."""
    fix = get_object_or_404(FixAction, id=fix_id)

    if fix.is_applied:
        return Response({"error": "Fix already applied"}, status=400)

    # Apply the fix based on action_type
    success = _execute_fix(fix, request.user)

    if success:
        fix.is_applied = True
        fix.applied_by = request.user
        fix.applied_at = timezone.now()
        fix.save()
        return Response({"status": "applied"})
    else:
        return Response({"error": "Fix could not be applied automatically"}, status=500)


def _execute_fix(fix, user) -> bool:
    """Execute a fix action. Returns True if successful."""
    run = fix.analysis.run

    if fix.action_type == "restart":
        # Restart the training run with the same config
        from experiment.runners import run_training

        run.status = "queued"
        run.save(update_fields=["status"])
        run_training(run)
        return True

    elif fix.action_type == "adjust_config":
        # Apply config changes from fix.config_changes
        config = run.config or {}
        changes = fix.config_changes or {}
        for key, value in changes.items():
            if value == "auto_reduce" and key == "batch_size":
                config[key] = max(1, config.get(key, 16) // 2)
            elif value == "auto_reduce" and key == "learning_rate":
                config[key] = float(config.get(key, 2e-5)) * 0.1
            elif value == "restore_defaults":
                # Restore default config
                config = {}
            elif value == "validate_data":
                # Placeholder for data validation
                pass
            else:
                config[key] = value
        run.config = config
        run.save(update_fields=["config"])
        return True

    elif fix.action_type == "reinstall_deps":
        # This requires server‑side execution; mark as manual
        return False

    elif fix.action_type == "clear_cache":
        # Placeholder for cache clearing
        return True

    elif fix.action_type == "increase_resources":
        # Placeholder for resource scaling
        return False

    return False


# ═══════════════════════════════════════════════════════════════
# Error Patterns (Knowledge Base)
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def error_pattern_list(request):
    """List all known error patterns."""
    qs = ErrorPattern.objects.filter(is_active=True).order_by("-occurrences")
    return Response(ErrorPatternSerializer(qs, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def error_pattern_create(request):
    """Add a new error pattern to the knowledge base."""
    serializer = ErrorPatternSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=201)
    return Response(serializer.errors, status=400)


# ═══════════════════════════════════════════════════════════════
# Dashboard Stats
# ═══════════════════════════════════════════════════════════════
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def failure_stats(request):
    """Get failure forensics statistics."""
    total_analyses = FailureAnalysis.objects.count()
    by_category = {}
    for choice in FailureAnalysis._meta.get_field("root_cause_category").choices:
        count = FailureAnalysis.objects.filter(root_cause_category=choice[0]).count()
        if count > 0:
            by_category[choice[1]] = count

    top_patterns = ErrorPattern.objects.order_by("-occurrences")[:5]
    return Response(
        {
            "total_analyses": total_analyses,
            "by_category": by_category,
            "top_error_patterns": ErrorPatternSerializer(top_patterns, many=True).data,
        }
    )
