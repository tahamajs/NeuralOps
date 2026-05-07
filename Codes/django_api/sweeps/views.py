from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from experiment.models import Sweep, Project, TrainingRun
from .serializers import SweepSerializer, SweepCreateSerializer
from .tasks import start_sweep


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def sweep_list(request):
    if request.method == 'GET':
        sweeps = Sweep.objects.select_related('project').order_by('-created_at')
        serializer = SweepSerializer(sweeps, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = SweepCreateSerializer(data=request.data)
        if serializer.is_valid():
            sweep = serializer.save()
            # Enqueue the Celery task that runs the sweep
            start_sweep.delay(sweep.id)
            return Response(SweepSerializer(sweep).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PATCH'])
@permission_classes([IsAuthenticated])
def sweep_detail(request, sweep_id):
    sweep = get_object_or_404(Sweep, id=sweep_id)

    if request.method == 'GET':
        serializer = SweepSerializer(sweep)
        return Response(serializer.data)

    elif request.method == 'PATCH':
        # Only allow status updates (pause / resume)
        new_status = request.data.get('status')
        if new_status in ['active', 'paused']:
            sweep.status = new_status
            sweep.save(update_fields=['status'])
            return Response(SweepSerializer(sweep).data)
        return Response({"error": "Invalid status"}, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def sweep_trials(request, sweep_id):
    sweep = get_object_or_404(Sweep, id=sweep_id)
    trials = sweep.trials.select_related('run').order_by('trial_number')
    from .serializers import SweepTrialSerializer
    serializer = SweepTrialSerializer(trials, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def sweep_importance(request, sweep_id):
    """Return parameter importance computed by Optuna."""
    sweep = get_object_or_404(Sweep, id=sweep_id)
    try:
        from .engine import get_parameter_importance
        importance = get_parameter_importance(sweep)
        return Response(importance)
    except Exception as e:
        return Response({"error": str(e)}, status=500)