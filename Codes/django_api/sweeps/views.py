from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .engine import (export_study, get_empirical_attainment_surface,
                     get_optimization_history, get_parallel_coordinate,
                     get_parameter_importance, get_pareto_front,
                     get_slice_plot, get_study_dataframe, get_trial_contour,
                     get_warm_start_config, import_study, report_intermediate,
                     should_prune)
from .models import Sweep, SweepTrial
from .serializers import (SweepCreateSerializer, SweepSerializer,
                          SweepTrialSerializer)
from .tasks import start_sweep


# ── Sweep CRUD ──────────────────────────────────────────────────────
@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def sweep_list(request):
    if request.method == "GET":
        sweeps = Sweep.objects.select_related("project").order_by("-created_at")
        serializer = SweepSerializer(sweeps, many=True)
        return Response(serializer.data)
    elif request.method == "POST":
        serializer = SweepCreateSerializer(data=request.data)
        if serializer.is_valid():
            sweep = serializer.save()
            start_sweep.delay(sweep.id)
            return Response(SweepSerializer(sweep).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def sweep_detail(request, sweep_id):
    sweep = get_object_or_404(Sweep, id=sweep_id)
    if request.method == "GET":
        serializer = SweepSerializer(sweep)
        return Response(serializer.data)
    elif request.method == "PATCH":
        new_status = request.data.get("status")
        if new_status in ["active", "paused"]:
            sweep.status = new_status
            sweep.save(update_fields=["status"])
            return Response(SweepSerializer(sweep).data)
        return Response({"error": "Invalid status"}, status=400)


# ── Trials ───────────────────────────────────────────────────────────
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_trials(request, sweep_id):
    sweep = get_object_or_404(Sweep, id=sweep_id)
    trials = sweep.trials.select_related("run").order_by("trial_number")
    serializer = SweepTrialSerializer(trials, many=True)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def trial_detail(request, sweep_id, trial_number):
    trial = get_object_or_404(SweepTrial, sweep_id=sweep_id, trial_number=trial_number)
    serializer = SweepTrialSerializer(trial)
    return Response(serializer.data)


# ── Preferential Optimization ───────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def submit_preference(request, sweep_id):
    """Submit a human preference between two trials."""
    sweep = get_object_or_404(Sweep, id=sweep_id)
    trial_a = request.data.get("trial_a")
    trial_b = request.data.get("trial_b")
    preference = request.data.get("preference")  # 'a' or 'b'
    if not sweep.preference_enabled:
        return Response({"error": "Preferential optimization not enabled"}, status=400)
    sweep.preference_queries.append(
        {
            "trial_a": trial_a,
            "trial_b": trial_b,
            "preference": preference,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
    )
    sweep.save(update_fields=["preference_queries"])
    return Response({"status": "recorded"})


# ── Intermediate Reporting ──────────────────────────────────────────
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def report_intermediate_value(request, sweep_id, trial_number):
    """Report an intermediate objective value for pruning."""
    sweep = get_object_or_404(Sweep, id=sweep_id)
    step = request.data.get("step", 0)
    value = request.data.get("value")
    if value is None:
        return Response({"error": "value required"}, status=400)
    report_intermediate(sweep, trial_number, step, value)
    pruned = should_prune(sweep, trial_number)
    return Response({"pruned": pruned})


# ── Visualizations ──────────────────────────────────────────────────
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_importance(request, sweep_id):
    return Response(get_parameter_importance(get_object_or_404(Sweep, id=sweep_id)))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_pareto(request, sweep_id):
    return Response(get_pareto_front(get_object_or_404(Sweep, id=sweep_id)))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_contour(request, sweep_id):
    params = request.query_params.getlist("params", [])
    return Response(get_trial_contour(get_object_or_404(Sweep, id=sweep_id), params))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_history(request, sweep_id):
    return Response(get_optimization_history(get_object_or_404(Sweep, id=sweep_id)))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_slice(request, sweep_id):
    param = request.query_params.get("param")
    if not param:
        return Response({"error": "param required"}, status=400)
    return Response(get_slice_plot(get_object_or_404(Sweep, id=sweep_id), param))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_parallel(request, sweep_id):
    return Response(get_parallel_coordinate(get_object_or_404(Sweep, id=sweep_id)))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_edf(request, sweep_id):
    return Response(
        get_empirical_attainment_surface(get_object_or_404(Sweep, id=sweep_id))
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_warm_start(request, sweep_id):
    return Response(get_warm_start_config(get_object_or_404(Sweep, id=sweep_id)))


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_dataframe(request, sweep_id):
    return Response(get_study_dataframe(get_object_or_404(Sweep, id=sweep_id)))


# ── Export / Import ─────────────────────────────────────────────────
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sweep_export(request, sweep_id):
    sweep = get_object_or_404(Sweep, id=sweep_id)
    path = f"/tmp/sweep_{sweep.id}.json"
    export_study(sweep, path)
    from django.http import FileResponse

    return FileResponse(
        open(path, "rb"), as_attachment=True, filename=f"sweep_{sweep.id}.json"
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def sweep_import(request):
    file = request.FILES.get("file")
    sweep_id = request.data.get("sweep_id")
    if not file or not sweep_id:
        return Response({"error": "file and sweep_id required"}, status=400)
    sweep = get_object_or_404(Sweep, id=sweep_id)
    path = f"/tmp/import_{sweep.id}.json"
    with open(path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    import_study(sweep, path)
    return Response({"status": "imported"})


# ── Trial Update (notes, user_attrs) ────────────────────────────────
@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def trial_update(request, sweep_id, trial_number):
    trial = get_object_or_404(SweepTrial, sweep_id=sweep_id, trial_number=trial_number)
    if "notes" in request.data:
        trial.notes = request.data["notes"]
    if "user_attrs" in request.data:
        trial.user_attrs = {**trial.user_attrs, **request.data["user_attrs"]}
    trial.save()
    return Response(SweepTrialSerializer(trial).data)
